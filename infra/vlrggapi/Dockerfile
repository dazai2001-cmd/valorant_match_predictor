FROM alpine:3.22 AS source

ARG VLRGGAPI_COMMIT=a6075fec9757ae5394c9a50002ba40270f4a5d00
WORKDIR /source
RUN apk add --no-cache curl \
    && curl -fsSL "https://github.com/axsddlr/vlrggapi/archive/${VLRGGAPI_COMMIT}.tar.gz" \
    | tar -xz --strip-components=1

FROM python:3.14.5-alpine AS builder

WORKDIR /vlrggapi
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /bin/
COPY --from=source /source/requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

FROM python:3.14.5-alpine

WORKDIR /vlrggapi
RUN addgroup -S vlrggapi \
    && adduser -S -G vlrggapi -h /vlrggapi -s /sbin/nologin vlrggapi

COPY --from=builder --chown=vlrggapi:vlrggapi /usr/local /usr/local
COPY --from=source --chown=vlrggapi:vlrggapi /source/api ./api
COPY --from=source --chown=vlrggapi:vlrggapi /source/models ./models
COPY --from=source --chown=vlrggapi:vlrggapi /source/routers ./routers
COPY --from=source --chown=vlrggapi:vlrggapi /source/utils ./utils
COPY --from=source --chown=vlrggapi:vlrggapi /source/main.py .

USER vlrggapi
EXPOSE 3001
CMD ["python", "main.py"]
HEALTHCHECK --interval=10s --timeout=5s --start-period=10s --retries=5 \
  CMD python -c "import urllib.request, json; r = urllib.request.urlopen('http://127.0.0.1:3001/v2/health', timeout=3); assert json.loads(r.read())['status'] == 'success'"

