import { ModuleRef } from '@nestjs/core';
import { IAITool, ToolHandler, IToolCall } from './types';

export class ToolManager {
    private toolHandlers: Map<string, ToolHandler> = new Map();
    private tools: IAITool[] = [];

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
        console.log({ toolHandlers });
        for (const key of Object.keys(toolHandlers)) {
            this.toolHandlers.set(key, toolHandlers[key]);
        }
    }

    async callTools<T = unknown, O = unknown>(tools: IToolCall[], moduleRef: ModuleRef, aiToolOptions?: O): Promise<T[]> {
        const results: T[] = [];
        for (const tool of tools) {
            if (this.toolHandlers.has(tool.function.name)) {
                const result = await this.callAiTool<T, O>(tool, moduleRef, aiToolOptions);
                if (result) {
                    results.push(result);
                }
            }
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
