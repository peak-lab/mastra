import { createTool } from '@mastra/core/tools';
import { MDocument } from '@mastra/rag';
import { openai } from '@ai-sdk/openai';
import { embedMany } from 'ai';
import { readFile } from 'fs/promises';
import { resolve } from 'path';
import { z } from 'zod';
import { PDFParse } from 'pdf-parse';

export const ingestPdfTool = createTool({
  id: 'ingest-pdf-tool',
  description: 'Ingest a PDF file into the knowledge base for RAG. Provide the absolute file path to the PDF.',
  inputSchema: z.object({
    filePath: z.string().describe('Absolute path to the PDF file to ingest'),
  }),
  outputSchema: z.object({
    success: z.boolean(),
    chunksIngested: z.number(),
    fileName: z.string(),
    message: z.string(),
  }),
  execute: async (inputData, context) => {
    const { filePath } = inputData;
    const fileName = filePath.split('/').pop() || filePath;

    try {
      const vectorStore = context?.mastra?.getVector('libsql');
      if (!vectorStore) {
        return { success: false, chunksIngested: 0, fileName, message: 'Vector store not available' };
      }

      const pdfBuffer = await readFile(resolve(filePath));
      const parser = new PDFParse(pdfBuffer);
      const pdfData = await parser.getText();
      const text = pdfData.text;

      if (!text.trim()) {
        return { success: false, chunksIngested: 0, fileName, message: 'PDF has no extractable text' };
      }

      await vectorStore.createIndex({
        indexName: 'knowledge',
        dimension: 1536,
      });

      const doc = MDocument.fromText(text);
      const chunks = await doc.chunk({
        strategy: 'recursive',
        maxSize: 512,
        overlap: 50,
      });

      const { embeddings } = await embedMany({
        values: chunks.map((chunk) => chunk.text),
        model: openai.embedding('text-embedding-3-small'),
      });

      await vectorStore.upsert({
        indexName: 'knowledge',
        vectors: embeddings,
        metadata: chunks.map((chunk) => ({
          text: chunk.text,
          source: fileName,
        })),
      });

      return {
        success: true,
        chunksIngested: chunks.length,
        fileName,
        message: `Successfully ingested ${chunks.length} chunks from ${fileName}`,
      };
    } catch (error) {
      return {
        success: false,
        chunksIngested: 0,
        fileName,
        message: `Error: ${error instanceof Error ? error.message : String(error)}`,
      };
    }
  },
});
