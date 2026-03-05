FROM node:22-slim AS base

RUN npm install -g bun

WORKDIR /app
COPY package.json bun.lock ./
RUN bun install --frozen-lockfile

COPY . .
RUN bun run build

RUN mkdir -p /app/data

ENV DATA_DIR=/app/data

EXPOSE 4111

CMD ["bun", "run", "start"]
