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

# Росздравнадзор использует российскую цепочку сертификатов Минцифры.
# Обычного Debian CA-bundle для неё недостаточно, поэтому добавляем
# корневой и выпускающие RSA-сертификаты в системное хранилище. Проверка TLS
# остаётся включённой — verify=False нигде не используется.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && mkdir -p /usr/local/share/ca-certificates/russian-trusted \
    && curl -fsSL https://gu-st.ru/content/lending/russian_trusted_root_ca_pem.crt \
        -o /usr/local/share/ca-certificates/russian-trusted/russian_trusted_root_ca.crt \
    && curl -fsSL https://gu-st.ru/content/lending/russian_trusted_sub_ca_pem.crt \
        -o /usr/local/share/ca-certificates/russian-trusted/russian_trusted_sub_ca.crt \
    && curl -fsSL https://gu-st.ru/content/lending/russian_trusted_sub_ca_2024_pem.crt \
        -o /usr/local/share/ca-certificates/russian-trusted/russian_trusted_sub_ca_2024.crt \
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
