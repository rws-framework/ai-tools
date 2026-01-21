import { ModuleRef } from '@nestjs/core';
import { IAITool, ToolHandler, IToolCall } from './types';

export class ToolManager {
    private toolHandlers: Map<string, ToolHandler> = new Map();
    private tools: IAITool[] = [];

    clearTools(): void {
        this.tools = [];
        this.toolHandlers.clear();
    }

     clearTool(toolName: string): void {
        this.tools = this.tools.filter(tool => tool.name !== toolName);
        this.toolHandlers.delete(toolName);
    }

    setTools(tools: IAITool[]): void {
        this.tools = tools;
    }

    getTools(): IAITool[] {
        return this.tools;
    }

    getTool(key: string): { definition: IAITool, handler: ToolHandler } | null {
        const foundTool = this.tools.find(tool => tool.name === key);

        if (foundTool) {
            return {
                definition: foundTool,
                handler: this.toolHandlers.get(foundTool.name)
            };
        }

        return null;
    }

    registerToolHandlers(toolHandlers: { [key: string]: ToolHandler }): void {        
        for (const key of Object.keys(toolHandlers)) {
            this.toolHandlers.set(key, toolHandlers[key]);
        }
    }

    async callTools<T = unknown, O = unknown>(tools: IToolCall[], moduleRef: ModuleRef, aiToolOptions?: O): Promise<T[]> {
        const results: T[] = [];
        const errors: Error[] = [];
        
        for (const tool of tools) {
            if (this.toolHandlers.has(tool.function.name)) {
                try {
                    const result = await this.callAiTool<T, O>(tool, moduleRef, aiToolOptions);
                    if (result) {
                        results.push(result);
                    }
                } catch (error) {
                    console.error(`Tool execution failed for ${tool.function.name}:`, error);
                    errors.push(error as Error);
                    // Continue with other tools instead of failing completely
                }
            } else {
                console.warn(`No handler found for tool: ${tool.function.name}`);
            }
        }
        
        // If all tools failed, throw the first error
        if (results.length === 0 && errors.length > 0) {
            throw errors[0];
        }

        return results;
    }

    private async callAiTool<T, O>(tool: IToolCall, moduleRef: ModuleRef, aiToolOptions?: O): Promise<T> {
        const handler = this.toolHandlers.get(tool.function.name);
        return await handler(tool.function.arguments, moduleRef, aiToolOptions);
    }

    getToolHandlers(): Map<string, ToolHandler> {
        return this.toolHandlers;
    }
}
