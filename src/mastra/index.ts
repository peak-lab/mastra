
import { resolve } from 'path';
import { Mastra } from '@mastra/core/mastra';
import { PinoLogger } from '@mastra/loggers';
import { LibSQLStore } from '@mastra/libsql';
import { LibSQLVector } from '@mastra/libsql';
import { Observability, DefaultExporter, CloudExporter, SensitiveDataFilter } from '@mastra/observability';

import { weatherAgent } from './agents/weather-agent';
import { ragAgent } from './agents/rag-agent';

const dbDir = resolve(import.meta.dirname, '..', '..');

export const mastra = new Mastra({
  agents: { weatherAgent, ragAgent },
  storage: new LibSQLStore({
    id: "mastra-storage",
    url: `file:${resolve(dbDir, 'mastra.db')}`,
  }),
  vectors: {
    libsql: new LibSQLVector({
      id: 'libsql-vector',
      url: `file:${resolve(dbDir, 'vector.db')}`,
    }),
  },
  logger: new PinoLogger({
    name: 'Mastra',
    level: 'info',
  }),
  observability: new Observability({
    configs: {
      default: {
        serviceName: 'mastra',
        exporters: [
          new DefaultExporter(),
          new CloudExporter(),
        ],
        spanOutputProcessors: [
          new SensitiveDataFilter(),
        ],
      },
    },
  }),
});
