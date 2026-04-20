import {execFile} from 'node:child_process';
import {promisify} from 'node:util';

const execFileAsync = promisify(execFile);

export type VizTask = {
	task_id: string;
	parent_task_id: string | null;
	agent_type: string | null;
	status: string;
	description: string;
	started_at: string | null;
	updated_at: string | null;
	finished_at: string | null;
	detached_at?: string | null;
	pid: number | null;
	event_count: number;
	path?: number[];
	lane: string;
};

export type VizEvent = {
	timestamp: string;
	task_id: string;
	agent_type: string | null;
	action: string;
	result: string;
	event_type: string;
	lane: string;
	summary: string;
};

export type VizConversation = {
	timestamp: string;
	role: string;
	text: string;
	kind: string;
	task_id?: string;
	lane: string;
};

export type VizSnapshot = {
	generated_at: string;
	session: {
		session_id: string | null;
		started_at: string | null;
		status: string;
	};
	summary: {
		task_count: number;
		running_count: number;
		decision_count: number;
	};
	root: {
		id: string;
		name: string;
		status: string;
	};
	tasks: VizTask[];
	events: VizEvent[];
	conversation: VizConversation[];
	recent_decisions: Array<{
		timestamp: string;
		input_summary: string;
		action: string;
		task_id?: string;
	}>;
};

export const loadSnapshot = async (
	projectRoot: string,
	pythonBin: string
): Promise<VizSnapshot> => {
	const {stdout} = await execFileAsync(
		pythonBin,
		['-m', 'mr1.viz', '--snapshot', '--project-root', projectRoot],
		{
			cwd: projectRoot,
			maxBuffer: 2 * 1024 * 1024
		}
	);

	return JSON.parse(stdout) as VizSnapshot;
};
