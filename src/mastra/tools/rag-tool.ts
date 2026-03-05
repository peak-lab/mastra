import { createVectorQueryTool } from '@mastra/rag';
import { openai } from '@ai-sdk/openai';

export const ragTool = createVectorQueryTool({
  id: 'rag-query-tool',
  vectorStoreName: 'libsql',
  indexName: 'knowledge',
  model: openai.embedding('text-embedding-3-small'),
  description: 'Search the knowledge base for relevant information to answer user questions.',
});
