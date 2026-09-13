"""Оркестрация подбора: поиск → скрейп → Supplier Gate → реестр → отчёт.

Широкий discovery допустим, но сырой поисковый источник не становится кандидатом
в отчёте, пока не подтверждена одновременно связь с товаром и роль поставщика.
"""
from __future__ import annotations
import asyncio, logging, re
from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any
from sqlalchemy.ext.asyncio import AsyncSession
from bot.db import repo
from bot.db.models import RegistryState, RequestStatus
from bot.db.repo import CandidateInput, SupplierInput
from bot.db.session import session_scope
from bot.logging_setup import log_extra
from bot.services import budget, guard
from bot.services import criteria as criteria_service
from bot.services.firecrawl import ScrapeResult, get_firecrawl_service
from bot.services.perplexity import get_perplexity_service
from bot.services.registry import RegistryResult, get_registry_service
from bot.services.report import CandidateView, Report, rank_candidates
logger=logging.getLogger(__name__)
MAX_SITES_TO_SCRAPE=24
PAGE_RU_RE=re.compile(r"\b(?:РЗН|ФСР|ФСЗ)\s*(?:№\s*)?\d{4}/\d+(?:[-/]\d+)?\b",re.I|re.UNICODE)
_GENERIC_MATCH_TERMS={"степлер","кожный","одноразовый","одноразовая","стерильный","стерильная","изделие","медицинский","медицинское","набор","система","инструмент","аппарат","устройство","скоба","скобы","скобами","штук","упаковка"}
_NON_SUPPLIER_MARKERS={"сертификат соответствия","сертификац","реестр сертификатов","44-фз","223-фз","закупки","тендер","фас россии","юридические услуги"}
_SUPPLIER_MARKERS={"купить","цена","в наличии","заказать","корзин","поставк","поставщик","дистрибьютор","дилер","коммерческое предложение","запросить цену","оставить заявку","медицинское оборудование","медицинские изделия"}
@dataclass(slots=True)
class SearchSummary:
 total_found:int=0; blacklisted:int=0; scraped:int=0; registry_state:str=RegistryState.UNAVAILABLE; unrega_state:str=RegistryState.UNAVAILABLE; errors:list[str]=field(default_factory=list); search_failed:bool=False; budget_exceeded:bool=False
async def run_search(*,request_id:int,product:str,requirements:list[str])->SearchSummary:
 extra=log_extra(request_id); summary=SearchSummary(); registry_service=get_registry_service(); registry=await registry_service.check_product(product,request_id=request_id,cache=True); best=registry.best
 search=await get_perplexity_service().find_suppliers(product,requirements=requirements,ru_number=best.ru_number if best else None,holder=best.holder if best else None,request_id=request_id); summary.registry_state=registry.state
 if not search.ok:
  summary.search_failed=True; summary.budget_exceeded=budget.exceeded(request_id); summary.errors.append(search.error or "поиск не удался"); return summary
 if registry.unavailable: summary.errors.append("реестр недоступен")
 summary.total_found=len(search.suppliers)
 if not search.suppliers: return summary
 tainted=set(); inputs=[]
 for item in search.suppliers:
  screening=await guard.screen_third_party_async(f"{item.name} {item.note}",source="выдача поиска")
  if screening.suspicious: tainted.add(item.site or item.name)
  inputs.append(SupplierInput(name=item.name,domain=item.site or None,email=item.email or None,phone=item.phone or None,found_via=search.query[:500]))
 async with session_scope() as session: key_to_id=await repo.upsert_suppliers(session,inputs)
 to_scrape=[x for x in search.suppliers if x.site][:MAX_SITES_TO_SCRAPE]
 scrape_task=get_firecrawl_service().scrape_many([x.site for x in to_scrape],product,request_id=request_id) if to_scrape else _nothing(); unrega_task=registry_service.check_unrega(product,holder=best.holder if best else None,request_id=request_id)
 results,unrega=await asyncio.gather(scrape_task,unrega_task); scrapes={r.url:r for r in results}; summary.scraped=sum(1 for r in results if r.ok); summary.unrega_state=unrega.state
 if unrega.unavailable: summary.errors.append("информационные письма не проверены")
 candidates=[]; accepted=[]; gated=0
 for item in search.suppliers:
  sid=_resolve_supplier_id(item,key_to_id)
  if sid is None: continue
  scrape=scrapes.get(item.site) if item.site else None; ru_match,basis=_site_registry_match(product,scrape,best); allowed,gate_basis=_supplier_gate(item,product,scrape,best,ru_match)
  if not allowed:
   gated+=1; logger.info("Supplier Gate: исключён %s — %s",item.site or item.name,gate_basis,extra=extra); continue
  accepted.append(item)
  evidence_url=(scrape.evidence_url if scrape and scrape.evidence_url else item.site) or None
  candidates.append(CandidateInput(supplier_id=sid,site_claims=scrape.claims_stock if scrape and scrape.ok else None,site_url=evidence_url,site_price=scrape.price if scrape and scrape.ok else None,ru_number=best.ru_number if best else None,ru_holder=best.holder if best else None,ru_valid=best.valid if best else None,ru_registry=best.registry if best else None,ru_checked_at=None if registry.unavailable else registry.checked_at,unrega_flags=_unrega_flags(unrega,ru_site_match=ru_match,ru_match_basis=basis),raw={"registry":registry.as_payload(),"registry_match":{"site_matches_ru":ru_match,"basis":basis},"supplier_gate":{"allowed":True,"basis":gate_basis},"search":{"note":item.note,"source":item.source_url},"scrape":{"ok":bool(scrape and scrape.ok),"error":scrape.error if scrape else None,"evidence_url":evidence_url,"injection_suspected":bool((scrape and scrape.injection_suspected) or (item.site or item.name) in tainted)}}))
 async with session_scope() as session:
  if candidates: await repo.upsert_candidates(session,request_id,candidates)
  await _enrich_contacts(session,accepted,scrapes,key_to_id); summary.blacklisted=await repo.count_blacklisted_in_request(session,request_id); await repo.transition(session,request_id,RequestStatus.REPORT)
 summary.budget_exceeded=budget.exceeded(request_id); logger.info("Подбор: discovery %s, проверено %s, Supplier Gate исключил %s, в отчёт %s",summary.total_found,summary.scraped,gated,len(candidates),extra=extra); return summary
async def close_request(session:AsyncSession,request_id:int)->bool:
 closed=await repo.transition(session,request_id,RequestStatus.CLOSED); budget.forget(request_id); return closed
async def _nothing()->list[ScrapeResult]: return []
def _normalise_ru(v:str)->str: return re.sub(r"[^a-zа-яё0-9]","",(v or "").lower())
def _distinctive_terms(text:str)->set[str]: return {w for w in re.findall(r"[A-Za-zА-Яа-яЁё0-9-]{4,}",(text or "").lower()) if w not in _GENERIC_MATCH_TERMS and not w.isdigit()}
def _holder_matches(item:object,best:Any)->bool:
 if best is None: return False
 holder=str(getattr(best,"holder","") or "").lower(); name=str(getattr(item,"name","") or "").lower(); domain=str(getattr(item,"site","") or "").lower(); terms=[x for x in re.findall(r"[a-zа-яё0-9]{5,}",holder) if x not in {"общество","ограниченной","ответственностью"}]
 return bool(terms and any(t in name or t in domain for t in terms))
def _supplier_gate(item:object,product:str,scrape:ScrapeResult|None,best:Any,ru_match:bool|None)->tuple[bool,str]:
 if _holder_matches(item,best): return True,"держатель/производитель РУ"
 if scrape is None or not scrape.ok or not scrape.markdown: return False,"страница не проверена"
 text=scrape.markdown.lower(); note=str(getattr(item,"note","") or "").lower(); commercial=bool(getattr(scrape,"claims_stock",None) is True or getattr(scrape,"price",None) is not None or any(m in text for m in _SUPPLIER_MARKERS)); negative=sum(1 for m in _NON_SUPPLIER_MARKERS if m in text); product_hit=ru_match is True or any(t in text for t in _distinctive_terms(product)); model_supplier=any(x in note for x in ("поставщик","продав","дистриб","дилер","производител"))
 if negative>=2 and not (getattr(scrape,"price",None) is not None or getattr(scrape,"claims_stock",None) is True): return False,"информационный/сертификационный/закупочный ресурс"
 if product_hit and commercial: return True,"товар + коммерческие признаки на сайте"
 if product_hit and model_supplier and (getattr(scrape,"email",None) or getattr(scrape,"phone",None)): return True,"товар + признаки поставщика + контакты"
 return False,"нет одновременного подтверждения товара и роли поставщика"
def _site_registry_match(product:str,scrape:ScrapeResult|None,best:Any)->tuple[bool|None,str]:
 if best is None or not getattr(best,"ru_number",None): return None,"РУ на изделие не найдено"
 if scrape is None or not scrape.ok or not scrape.markdown: return None,"страница поставщика не проверена"
 page=scrape.markdown.lower(); expected=_normalise_ru(str(best.ru_number)); compact=_normalise_ru(page)
 if expected and expected in compact: return True,"точный номер РУ найден на странице поставщика"
 nums={_normalise_ru(m.group(0)) for m in PAGE_RU_RE.finditer(scrape.markdown)}; nums.discard("")
 if nums and expected not in nums: return False,"на странице поставщика указан другой номер РУ"
 raw=getattr(best,"raw",{}) or {}; raw_text=raw.get("text") if isinstance(raw,dict) else ""; registry_text=" ".join(str(v or "") for v in (getattr(best,"product_name",None),getattr(best,"holder",None),raw_text)).lower(); strong={t for t in _distinctive_terms(product) if t in registry_text}; matched={t for t in strong if t in page}
 if matched: return None,"совпал бренд/модель ("+", ".join(sorted(matched)[:3])+") но номер РУ на странице не найден"
 return None,"на странице поставщика недостаточно данных для привязки к РУ"
def _unrega_flags(unrega:RegistryResult,*,ru_site_match:bool|None,ru_match_basis:str)->dict[str,Any]: return {"state":unrega.state,"items":[r.product_name or r.status_text or "письмо" for r in unrega.records],"errors":unrega.errors,"ru_site_match":ru_site_match,"ru_match_basis":ru_match_basis}
def _resolve_supplier_id(item:object,key_to_id:dict[str,int])->int|None:
 site=getattr(item,"site","") or ""; name=getattr(item,"name","") or ""
 if site:
  domain=repo.normalise_domain(site)
  if domain and domain in key_to_id: return key_to_id[domain]
 return key_to_id.get(name.strip())
async def _enrich_contacts(session:AsyncSession,suppliers:Sequence[object],scrapes:dict[str,ScrapeResult],key_to_id:dict[str,int])->None:
 updates=[]
 for item in suppliers:
  site=getattr(item,"site","") or ""; scrape=scrapes.get(site)
  if scrape and scrape.ok and (scrape.email or scrape.phone): updates.append(SupplierInput(name=str(getattr(item,"name","")),domain=site or None,email=scrape.email,phone=scrape.phone))
 if updates: await repo.upsert_suppliers(session,updates)
async def build_report(session:AsyncSession,*,request_id:int,product:str,qty:str,requirements:list[str])->Report:
 request=await repo.get_request(session,request_id); token=request.token if request else "?"; rows=await repo.list_candidates_for_report(session,request_id); views=[CandidateView(candidate_id=int(r.id),supplier_id=int(r.supplier_id or 0),supplier_name=str(r.supplier_name),domain=r.domain,email=r.email,phone=r.phone,site_claims=r.site_claims,site_url=r.site_url,site_price=r.site_price if isinstance(r.site_price,Decimal) else None,ru_number=r.ru_number,ru_holder=r.ru_holder,ru_valid=r.ru_valid,ru_registry=r.ru_registry,registry_state=_registry_state(r),ru_site_match=_ru_site_match(r.unrega_flags),ru_match_basis=_ru_match_basis(r.unrega_flags),unrega_flags=_unrega_items(r.unrega_flags),injection_suspected=bool(r.injection_suspected)) for r in rows]; unrega_state=_unrega_state(rows); known=await criteria_service.for_prompt(session); ordered,summary,missing,failed=await rank_candidates(views,product=product,qty=qty,requirements=requirements,criteria=known,request_id=request_id,unrega_state=unrega_state); await repo.set_candidate_ranks(session,{v.candidate_id:i for i,v in enumerate(ordered,1)}); await repo.transition(session,request_id,RequestStatus.AWAITING_CHOICE); return Report(request_token=token,product=product,candidates=ordered,summary=summary,missing_data=missing,llm_failed=failed,unrega_state=unrega_state)
def _registry_state(r:object)->str:
 s=getattr(r,"registry_state",None)
 if s in (RegistryState.FOUND,RegistryState.NOT_FOUND,RegistryState.UNAVAILABLE): return str(s)
 return RegistryState.FOUND if getattr(r,"ru_number",None) else RegistryState.UNAVAILABLE
def _unrega_items(v:object)->list[str]: return [str(x) for x in v.get("items",[]) or []] if isinstance(v,dict) else []
def _ru_site_match(v:object)->bool|None:
 m=v.get("ru_site_match") if isinstance(v,dict) else None; return m if isinstance(m,bool) else None
def _ru_match_basis(v:object)->str|None:
 b=v.get("ru_match_basis") if isinstance(v,dict) else None; return str(b) if b else None
def _unrega_state(rows:Sequence[Any])->str:
 for r in rows:
  f=getattr(r,"unrega_flags",None)
  if isinstance(f,dict) and f.get("state") in (RegistryState.FOUND,RegistryState.NOT_FOUND,RegistryState.UNAVAILABLE): return str(f["state"])
 return RegistryState.UNAVAILABLE if rows else RegistryState.NOT_FOUND
