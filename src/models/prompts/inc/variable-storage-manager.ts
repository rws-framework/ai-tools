export class VariableStorageManager {
    private varStorage: any = {};

    getVar<T>(key: string): T {
        return Object.keys(this.varStorage).includes(key) ? this.varStorage[key] : null;
    }

    setVar<T>(key: string, val: T): void {
        this.varStorage[key] = val;
    }

    clearVar(key: string): void {
        delete this.varStorage[key];
    }

    clearAllVars(): void {
        this.varStorage = {};
    }

    getAllVars(): any {
        return { ...this.varStorage };
    }

    hasVar(key: string): boolean {
        return Object.keys(this.varStorage).includes(key);
    }
}
