# ---------------------------------------------------------------------------
# Образ бота. Chromium для сборки КП живёт внутри этого же образа — отдельным
# сервисом не выносим.
# ---------------------------------------------------------------------------

FROM python:3.12-slim-bookworm AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PLAYWRIGHT_BROWSERS_PATH=/opt/pw-browsers \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
WORKDIR /app

FROM base AS builder
RUN python -m venv "$VIRTUAL_ENV"
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM base AS runtime

# Системный CA-bundle нужен в том числе для государственных HTTPS-сервисов
# с цепочкой сертификатов, которую минимальный Python-образ сам не содержит.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Системные зависимости Chromium ставит сам playwright (--with-deps).
COPY --from=builder /opt/venv /opt/venv
RUN playwright install --with-deps chromium \
    && rm -rf /var/lib/apt/lists/* /root/.cache

COPY bot/ ./bot/
COPY prompts/ ./prompts/
COPY skills/ ./skills/
COPY alembic.ini ./
COPY scripts/ ./scripts/

RUN groupadd --system app \
    && useradd --system --gid app --create-home --home-dir /home/app app \
    && chown -R app:app /app /opt/pw-browsers \
    && chmod +x scripts/*.sh
USER app

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD ["python", "-m", "bot.health"]

CMD ["python", "-m", "bot.main"]
