import { Injectable } from '@nestjs/common';
import PQueue from 'p-queue';
import { IBatchMetadata, IRateLimitConfig } from '../types/rag.types';
import tiktoken from 'tiktoken';
import { BlackLogger } from '@rws-framework/server/nest';

let encoding_for_model: any = null;
encoding_for_model = tiktoken.encoding_for_model

@Injectable()
export class OpenAIRateLimitingService {
    static readonly DEFAULT_CONFIG: Required<IRateLimitConfig> = {
        rpm: 500,
        tpm: 300_000,
        concurrency: 4,
        maxRetries: 6,
        baseBackoffMs: 500,
        safetyFactor: 0.75
    };

    private tokenizer: any = null;
    private queue: PQueue;
    private config: Required<IRateLimitConfig>;

    private logger = new BlackLogger(OpenAIRateLimitingService.name);

    constructor() {
        this.config = { ...OpenAIRateLimitingService.DEFAULT_CONFIG };
        this.queue = new PQueue({ concurrency: this.config.concurrency });
    }

    /**
     * Initialize the service with a specific model and configuration
     */
    initialize(model: string, config?: Partial<IRateLimitConfig>): void {
        if (config) {
            this.config = { ...this.config, ...config };
        }

        // Initialize tokenizer for precise token counting
        try {
            if (encoding_for_model) {
                this.tokenizer = encoding_for_model(model);
            } else {
                this.tokenizer = null;
            }
        } catch (e) {
            this.logger.warn(`Could not load tokenizer for model ${model}, using character-based estimation`);
            this.tokenizer = null;
        }

        // Reinitialize queue with new concurrency
        this.queue = new PQueue({ concurrency: this.config.concurrency });
    }

    /**
     * Execute a batch of operations with sophisticated rate limiting
     */
    async executeWithRateLimit<T, R>(
        items: T[],
        executor: (batch: T[]) => Promise<R[]>,
        tokenExtractor?: (item: T) => string
    ): Promise<R[]> {
        const tokensPerMinutePerWorker = Math.floor(
            this.config.tpm * this.config.safetyFactor / this.config.concurrency
        );
        const maxTokensPerCall = Math.max(1_000, Math.floor(tokensPerMinutePerWorker / 60));

        // Build batches based on token limits
        const batches = this.tokenizer && tokenExtractor
            ? Array.from(this.chunkByToken(items, maxTokensPerCall, tokenExtractor))
            : this.fallbackChunking(items, 128);

        // Map batch -> start index for placing results back
        const batchStarts: Array<IBatchMetadata<T>> = [];
        let idx = 0;
        for (const batch of batches) {
            batchStarts.push({ start: idx, batch });
            idx += batch.length;
        }

        const results = new Array(items.length);

        // Process all batches with queue concurrency control
        await Promise.all(batchStarts.map(meta =>
            this.queue.add(async () => {
                // Adaptive shrink loop on repeated 429s
                let attemptBatch = meta.batch;
                for (let attempt = 0; attempt < 6; attempt++) {
                    try {
                        const batchResults = await this.callWithRetry(() => executor(attemptBatch));
                        for (let i = 0; i < batchResults.length; i++) {
                            results[meta.start + i] = batchResults[i];
                        }
                        break;
                    } catch (err: any) {
                        const status = err?.status || err?.response?.status;
                        if (status === 429) {
                            // Shrink batch if >1 and retry quickly (binary shrink)
                            if (attemptBatch.length <= 1) throw err;
                            attemptBatch = attemptBatch.slice(0, Math.ceil(attemptBatch.length / 2));
                            this.logger.debug(`Rate limit hit, shrinking batch to ${attemptBatch.length} items`);
                            // Small sleep to avoid immediate retry stampede
                            await this.sleep(200 + Math.random() * 200);
                            continue;
                        }
                        throw err;
                    }
                }
            })
        ));

        await this.queue.onIdle();
        return results;
    }

    /**
     * Count tokens in text
     */
    private countTokens(text: string): number {
        if (!this.tokenizer) {
            return Math.ceil(text.length / 4); // Conservative fallback
        }
        return this.tokenizer.encode(text).length;
    }

    /**
     * Chunk items by token budget
     */
    private *chunkByToken<T>(
        items: T[], 
        maxTokensPerCall: number, 
        tokenExtractor: (item: T) => string
    ): Generator<T[]> {
        let batch: T[] = [];
        let tokens = 0;

        for (const item of items) {
            const text = tokenExtractor(item);
            const itemTokens = this.countTokens(text);
            
            if (batch.length && tokens + itemTokens > maxTokensPerCall) {
                yield batch;
                batch = [];
                tokens = 0;
            }
            
            batch.push(item);
            tokens += itemTokens;
        }

        if (batch.length) {
            yield batch;
        }
    }

    /**
     * Fallback chunking when tokenizer is not available
     */
    private fallbackChunking<T>(items: T[], itemsPerBatch: number): T[][] {
        const result: T[][] = [];
        for (let i = 0; i < items.length; i += itemsPerBatch) {
            result.push(items.slice(i, i + itemsPerBatch));
        }
        return result;
    }

    /**
     * Call function with retry logic and exponential backoff
     */
    private async callWithRetry<T>(fn: () => Promise<T>, attempt: number = 0): Promise<T> {
        try {
            return await fn();
        } catch (err: any) {
            const status = err?.status || err?.response?.status;
            const isRateLimit = status === 429 || (status >= 500 && status < 600);

            if (!isRateLimit || attempt >= this.config.maxRetries) {
                throw err;
            }

            const delay = Math.min(60_000, this.config.baseBackoffMs * (2 ** attempt));
            const jitter = Math.random() * 300;

            this.logger.warn(`Retrying request in ${delay + jitter}ms (attempt ${attempt + 1}/${this.config.maxRetries})`);
            await this.sleep(delay + jitter);

            return this.callWithRetry(fn, attempt + 1);
        }
    }

    /**
     * Sleep utility for delays
     */
    private sleep(ms: number): Promise<void> {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    /**
     * Get current configuration
     */
    getConfig(): Required<IRateLimitConfig> {
        return { ...this.config };
    }

    /**
     * Update configuration
     */
    updateConfig(newConfig: Partial<IRateLimitConfig>): void {
        this.config = { ...this.config, ...newConfig };
        
        // Update queue concurrency if changed
        if (newConfig.concurrency && newConfig.concurrency !== this.queue.concurrency) {
            this.queue = new PQueue({ concurrency: this.config.concurrency });
        }
    }
}