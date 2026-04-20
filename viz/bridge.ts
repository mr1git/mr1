import {spawn, type ChildProcessWithoutNullStreams} from 'node:child_process';
import process from 'node:process';
import type {VizSnapshot} from './snapshot.js';

export type BridgeEventPayload = {
	type: string;
	timestamp?: string;
	role?: string;
	text?: string;
	kind?: string;
	task_id?: string;
	parent_task_id?: string;
	agent_type?: string;
	status?: string;
	pid?: number;
	lane?: string;
	summary?: string;
	description?: string;
};

export type BridgeMessage =
	| {type: 'ready'; session_id: string}
	| {type: 'snapshot'; snapshot: VizSnapshot}
	| {type: 'event'; event: BridgeEventPayload}
	| {type: 'response'; text: string}
	| {type: 'command_result'; command: string; output: string}
	| {type: 'shutdown'; killed: number}
	| {type: 'error'; message: string; fatal?: boolean};

export class BridgeClient {
	private child: ChildProcessWithoutNullStreams | null = null;
	private buffer = '';
	private closed = false;

	constructor(
		private readonly projectRoot: string,
		private readonly pythonBin: string,
		private readonly onMessage: (message: BridgeMessage) => void,
		private readonly extraEnv: Record<string, string> = {}
	) {}

	start(): void {
		this.closed = false;
		this.child = spawn(this.pythonBin, ['-m', 'mr1.ui_bridge'], {
			cwd: this.projectRoot,
			env: {
				...process.env,
				MR1_PROJECT_ROOT: this.projectRoot,
				...this.extraEnv
			},
			stdio: ['pipe', 'pipe', 'pipe']
		});

		this.child.stdout.on('data', chunk => {
			this.buffer += chunk.toString('utf8');
			this.flushBuffer();
		});

		this.child.stderr.on('data', chunk => {
			this.onMessage({
				type: 'error',
				message: chunk.toString('utf8').trim()
			});
		});

		this.child.on('error', error => {
			this.onMessage({
				type: 'error',
				message: `Failed to start MR1 bridge: ${error.message}`,
				fatal: true
			});
		});

		this.child.on('exit', code => {
			if (!this.closed && code && code !== 0) {
				this.onMessage({
					type: 'error',
					message: `MR1 bridge exited with code ${code}`,
					fatal: true
				});
			}
		});
	}

	private flushBuffer(): void {
		while (true) {
			const newline = this.buffer.indexOf('\n');
			if (newline === -1) {
				return;
			}

			const line = this.buffer.slice(0, newline).trim();
			this.buffer = this.buffer.slice(newline + 1);
			if (!line) {
				continue;
			}

			try {
				this.onMessage(JSON.parse(line) as BridgeMessage);
			} catch (error) {
				const message = error instanceof Error ? error.message : String(error);
				this.onMessage({type: 'error', message: `Bridge parse error: ${message}`});
			}
		}
	}

	send(payload: Record<string, unknown>): void {
		if (!this.child?.stdin.writable) {
			return;
		}
		this.child.stdin.write(`${JSON.stringify(payload)}\n`);
	}

	sendInput(text: string): void {
		this.send({type: 'input', text});
	}

	requestSnapshot(): void {
		this.send({type: 'snapshot'});
	}

	close(): void {
		if (!this.child) {
			return;
		}
		this.closed = true;
		if (this.child.stdin.writable) {
			this.send({type: 'shutdown'});
			this.child.stdin.end();
		}
	}
}
