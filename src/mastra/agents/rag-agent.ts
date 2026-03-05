import { Agent } from '@mastra/core/agent';
import { Memory } from '@mastra/memory';
import { createOpenRouter } from '@openrouter/ai-sdk-provider';
import { ragTool } from '../tools/rag-tool';
import { ingestPdfTool } from '../tools/ingest-pdf-tool';

const openrouter = createOpenRouter({
  apiKey: process.env.OPENROUTER_API_KEY,
});

export const ragAgent = new Agent({
  id: 'rag-agent',
  name: 'RAG Agent',
  instructions: `
    You are a knowledgeable assistant that can ingest PDF documents and answer questions based on your knowledge base.

    You have two capabilities:
    1. INGEST PDFs: When a user provides a PDF file path, use the ingest-pdf-tool to add it to the knowledge base.
    2. ANSWER QUESTIONS: When a user asks a question, use the rag-query-tool to search the knowledge base and answer based on the retrieved context.

    Rules:
    - Always cite which document or section your answer comes from when possible
    - If the context doesn't contain enough information, say so clearly
    - Be concise and accurate
    - When ingesting a PDF, confirm the number of chunks added
  `,
  model: openrouter('openai/gpt-4o'),
  tools: { ragTool, ingestPdfTool },
  memory: new Memory(),
});
