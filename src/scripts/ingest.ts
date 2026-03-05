import { mastra } from '../mastra/index';
import { MDocument } from '@mastra/rag';
import { openai } from '@ai-sdk/openai';
import { embedMany } from 'ai';

const docs = [
  {
    title: 'Mastra Overview',
    content: `Mastra is a TypeScript framework for building AI agents, workflows, and tools.
Users can initialize projects using npm, pnpm, Yarn, or Bun with single commands. The framework enables immediate testing through Studio.
Mastra supports integration with Next.js, React, Astro, Express, SvelteKit, and Hono, allowing developers to add it to existing projects or create new applications.

Conversational Agents maintain context across sessions using thread-based message history and observational memory — a system that compresses conversation histories into dense logs while preserving long-term recall. Agents stream responses token-by-token and can execute tools mid-conversation.

Domain-Specific Copilots are assistants grounded in user data through RAG pipelines featuring chunking, embedding, vector storage across 12+ providers, metadata filtering, and re-ranking. They support dynamic instructions and voice interaction via 12+ speech providers.

Workflow Automations are type-safe, multi-step processes with sequential chaining, parallel fan-out, conditional branching, loops, and concurrency control. Workflows suspend for human approval and resume execution.

Decision-Support Tools are systems analyzing data through composed tools querying databases and APIs, using Zod-validated structured outputs with supervisor coordination patterns.

AI-Powered Applications orchestrate multiple agents across Node.js runtimes, cloud providers, and frameworks with production monitoring and 10+ storage backend options.`,
  },
  {
    title: 'Mastra Agents',
    content: `Agents use LLMs and tools to solve open-ended tasks. They reason about goals, decide which tools to use, retain conversation memory, and iterate internally until the model emits a final answer or an optional stop condition is met.

Mastra supports more than 600 models. Agents are created by instantiating the Agent class with system instructions and a model parameter. Agents should be registered in a Mastra instance to make it available throughout your application.

Instructions define the agent's behavior, personality, and capabilities. Instructions can be provided as strings, arrays of strings, or arrays of system message objects. Provider-specific options like prompt caching and reasoning configuration can be set via providerOptions.

Response Generation: Agents support both generation (returning full output) and streaming (delivering tokens in real time). Generate works for internal responses while Stream delivers content to end users quickly.

Structured Output: Using Zod or JSON Schema, agents can return type-safe data with results available on response.object.

Image Analysis: Agents can process images by passing objects with type: 'image' and image URLs in the content array.

Tools Integration: Agents can access external APIs and services through registered tools, enabling structured interactions beyond language generation.

Multi-Step Operations: The maxSteps parameter controls maximum sequential LLM calls. The default is 1 but can be increased.

Subagents: Agents can have registered subagents, making the parent agent a supervisor that can delegate tasks to subagents.

Dynamic Instructions: Instructions can be async functions, enabling runtime resolution for personalization and A/B testing.

Studio provides workflow testing interface displaying execution lifecycle, step inputs/outputs, and detailed inspection capabilities for comprehensive workflow analysis. Agents are referenced via mastra.getAgent(), which is preferred over a direct import, since it provides access to the Mastra instance configuration.`,
  },
  {
    title: 'Mastra RAG (Retrieval-Augmented Generation)',
    content: `Mastra's RAG system enhances language model responses by incorporating relevant context from your own data sources, improving accuracy and grounding responses in real information.

The platform offers standardized APIs to process and embed documents alongside support for multiple vector storage solutions. It includes chunking and embedding strategies for optimal retrieval and provides observability for tracking embedding and retrieval performance.

The RAG process follows five steps:
1. Document Initialization: Convert text into MDocument objects
2. Chunking: Break documents into manageable pieces using strategies like recursive chunking (512-character size with 50-character overlap)
3. Embeddings: Generate vector representations using embedding models like OpenAI's text-embedding-3-small
4. Storage: Persist embeddings in vector databases via upsert operations
5. Retrieval: Query the vector database with embedding vectors to find similar content

Supported Vector Databases: pgvector, Pinecone, Qdrant, MongoDB, LibSQL, Chroma, LanceDB, and more.
Document Processing supports various strategies (recursive, sliding window, etc.) for chunking documents with metadata enrichment capabilities.

Retrieval: Mastra's retrieval system converts user queries to embeddings using the same model as documents, compares embeddings via vector similarity, and retrieves the most relevant chunks. Results can be filtered by metadata, re-ranked, or processed through knowledge graphs.

Basic Semantic Search uses vector similarity to find semantically comparable chunks. Users convert queries to embeddings, then query the vector store with a topK parameter specifying maximum results. Results include text content, similarity scores (0-1 range), and metadata fields.

Advanced Metadata Filtering supports hybrid vector search combining semantic matching with structured filters using MongoDB-style syntax: equality filters, numeric comparisons, multiple conditions and logical operators ($or, $and), and array operations.

The createVectorQueryTool function empowers agents to autonomously manage retrieval decisions. Supported databases include Pinecone, pgVector, Chroma, LanceDB, and others, each with specialized configurations.

Re-ranking improves initial vector search results by considering word order, exact matches, and cross-attention between queries and documents.`,
  },
  {
    title: 'Mastra Workflows',
    content: `Workflows in Mastra enable structured task sequences through defined steps rather than relying on single-agent reasoning. They offer precise control over task breakdown, data movement, and execution conditions, running via built-in engine or external runners like Inngest.

Use workflows for tasks that are clearly defined upfront and involve multiple steps with a specific execution order. They suit scenarios requiring fine-grained control over data transformation and primitive execution at each stage.

Three key concepts guide Mastra workflows:
1. Creating steps with createStep, defining input/output schemas and logic
2. Composing steps via createWorkflow to establish execution flow
3. Running workflows with built-in suspension, resumption, and streaming capabilities

Steps form workflow components. Using createStep() requires specifying inputSchema and outputSchema. The execute function determines functionality — calling internal functions, external APIs, agents, or tools.

Construct workflows using createWorkflow() with matching input/output schemas. Chain steps via .then() and finalize with .commit(). Control flow methods determine step schema structures.

Workflow state lets you share values across steps without passing them through every step's inputSchema and outputSchema. This facilitates tracking progress, accumulating results, or distributing configuration across entire workflows.

Nested Workflows function as reusable steps within larger compositions. Use cloneWorkflow() for logic reuse with independent tracking under distinct IDs.

Register workflows in Mastra instances for application-wide availability. Access them via getWorkflow() on mastra or mastraClient instances.

Two execution modes exist: Start (waits for completion before returning final results) and Stream (emits events during execution, enabling progress monitoring).

Workflow results return discriminated unions with status values: success, failed, suspended, tripwire, or paused.`,
  },
  {
    title: 'Mastra Agent Memory',
    content: `Agents require memory mechanisms to maintain context across interactions, since LLMs are stateless. Mastra agents support message history storage with optional features like working memory, semantic recall, and observational memory.

Memory is beneficial for multi-turn conversations that reference prior exchanges and building context within conversation threads. It's unnecessary for single-turn, independent interactions.

Memory functionality requires the @mastra/memory package plus a storage provider like @mastra/libsql. Memory is enabled by creating a Memory instance and passing it to an agent's memory option.

Message History Management uses two key fields:
- Resource: A stable identifier for the user or entity
- Thread: An ID isolating a specific conversation session

These enable persistent, thread-aware memory where agents can store and retrieve context using matching resource and thread values.

Observational Memory solves context window degradation for extended conversations by running background agents that compress old messages into dense observations. This preserves long-term memory while keeping context windows manageable. Configuration supports custom models and customizable thresholds.

Dynamic Memory Selection via RequestContext allows conditional selection of different memory or storage configurations based on request-specific values, enabling tier-based memory strategies.`,
  },
];

async function ingest() {
  const vectorStore = mastra.getVector('libsql');

  await vectorStore.createIndex({
    indexName: 'knowledge',
    dimension: 1536,
  });

  let totalChunks = 0;

  for (const doc of docs) {
    const mdoc = MDocument.fromText(doc.content);

    const chunks = await mdoc.chunk({
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
        source: doc.title,
      })),
    });

    console.log(`  [${doc.title}] ${chunks.length} chunks`);
    totalChunks += chunks.length;
  }

  console.log(`\nTotal: ${totalChunks} chunks ingested into the knowledge base.`);
}

ingest().catch(console.error);
