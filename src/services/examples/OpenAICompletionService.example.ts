/**
 * Example usage of OpenAIRateLimitingService for other AI operations
 * 
 * This demonstrates how to use the rate limiting service for:
 * - OpenAI completions
 * - Image generation
 * - Any other OpenAI API calls that need rate limiting
 */

import { OpenAIRateLimitingService, IRateLimitConfig } from '../OpenAIRateLimitingService';
import { OpenAI } from 'openai';

export class OpenAICompletionService {
    private rateLimitingService: OpenAIRateLimitingService;
    private openai: OpenAI;

    constructor(apiKey: string, config?: Partial<IRateLimitConfig>) {
        this.openai = new OpenAI({ apiKey });
        this.rateLimitingService = new OpenAIRateLimitingService();
        
        // Initialize with model-specific limits
        this.rateLimitingService.initialize('gpt-4', {
            rpm: 500,        // Adjust based on your OpenAI plan
            tpm: 30_000,     // Tokens per minute for GPT-4
            concurrency: 3,  // Lower concurrency for completion models
            maxRetries: 5,
            ...config
        });
    }

    /**
     * Generate completions with rate limiting
     */
    async generateCompletions(
        prompts: string[], 
        model: string = 'gpt-4-turbo'
    ): Promise<string[]> {
        return await this.rateLimitingService.executeWithRateLimit(
            prompts,
            async (batch: string[]) => {
                // Execute batch of completion requests
                const promises = batch.map(prompt => 
                    this.openai.chat.completions.create({
                        model,
                        messages: [{ role: 'user', content: prompt }],
                        max_tokens: 500
                    })
                );
                
                const results = await Promise.all(promises);
                return results.map(result => 
                    result.choices[0]?.message?.content || ''
                );
            },
            (prompt: string) => prompt // Token extractor for accurate batching
        );
    }

    /**
     * Generate images with rate limiting  
     */
    async generateImages(prompts: string[]): Promise<string[]> {
        return await this.rateLimitingService.executeWithRateLimit(
            prompts,
            async (batch: string[]) => {
                const promises = batch.map(prompt =>
                    this.openai.images.generate({
                        model: 'dall-e-3',
                        prompt,
                        size: '1024x1024',
                        quality: 'standard',
                        n: 1
                    })
                );
                
                const results = await Promise.all(promises);
                return results.map(result => 
                    result.data[0]?.url || ''
                );
            },
            (prompt: string) => prompt
        );
    }

    /**
     * Update rate limiting configuration
     */
    updateRateLimits(config: Partial<IRateLimitConfig>): void {
        this.rateLimitingService.updateConfig(config);
    }
}

/**
 * Usage example:
 * 
 * const completionService = new OpenAICompletionService(process.env.OPENAI_API_KEY, {
 *     rpm: 100,     // Lower RPM for your plan
 *     tpm: 10_000,  // Lower TPM 
 *     concurrency: 2
 * });
 * 
 * const prompts = [
 *     "Explain quantum computing",
 *     "Write a haiku about AI", 
 *     "Summarize the history of computing"
 * ];
 * 
 * const completions = await completionService.generateCompletions(prompts);
 * console.log(completions);
 */