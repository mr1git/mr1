import type {VizConversation, VizSnapshot, VizTask} from './snapshot.js';

export type PaletteEntry = {
	letter: string;
	color: string;
	label: string;
};

export type CameraState = {
	follow: boolean;
	panSeconds: number;
	secondsPerCell: number;
};

export const ROYGBIV: PaletteEntry[] = [
	{letter: 'R', color: '#ff453a', label: 'Red'},
	{letter: 'O', color: '#ff9f0a', label: 'Orange'},
	{letter: 'Y', color: '#ffd60a', label: 'Yellow'},
	{letter: 'G', color: '#30d158', label: 'Green'},
	{letter: 'B', color: '#0a84ff', label: 'Blue'},
	{letter: 'I', color: '#5e5ce6', label: 'Indigo'},
	{letter: 'V', color: '#bf5af2', label: 'Violet'}
];

export const colorForPath = (path: number[] = []): PaletteEntry => {
	if (path.length === 0) {
		return ROYGBIV[0];
	}
	const index = path.reduce((sum, part, depth) => sum + (part + 1) * (depth + 1), 0);
	return ROYGBIV[Math.min(index, ROYGBIV.length - 1)];
};

export const dimColor = (hex: string): string => {
	const value = hex.replace('#', '');
	const channels = [0, 2, 4].map(offset => Number.parseInt(value.slice(offset, offset + 2), 16));
	const next = channels
		.map(channel => Math.max(0, Math.min(255, Math.round(channel * 0.48))))
		.map(channel => channel.toString(16).padStart(2, '0'))
		.join('');
	return `#${next}`;
};

export const getCameraCenterMs = (nowMs: number, camera: CameraState): number =>
	nowMs + camera.panSeconds * 1000;

export const worldToColumn = (
	worldMs: number,
	nowMs: number,
	camera: CameraState,
	width: number
): number => {
	const center = getCameraCenterMs(nowMs, camera);
	const offsetSeconds = (worldMs - center) / 1000;
	return Math.round(width / 2 + offsetSeconds / camera.secondsPerCell);
};

export const getTaskHeadTimeMs = (task: VizTask, nowMs: number): number => {
	if (task.status === 'running') {
		return nowMs;
	}
	const iso = task.finished_at ?? task.updated_at ?? task.started_at;
	return iso ? new Date(iso).getTime() : nowMs;
};

export const isTaskLive = (task: VizTask): boolean => task.status === 'running';

export const buildTaskRowMap = (tasks: VizTask[], axisRow: number, height: number): Map<string, number> => {
	const map = new Map<string, number>();
	const upperLimit = 1;
	const lowerLimit = Math.max(upperLimit + 1, height - 2);
	for (const task of tasks) {
		const path = task.path ?? [];
		const branchMagnitude = path.reduce((sum, part, depth) => sum + (part + 1) * (depth + 1), 0);
		const direction = task.lane === 'system' ? -1 : 1;
		const rawRow = axisRow + direction * (2 + branchMagnitude * 2);
		const clamped = Math.max(upperLimit, Math.min(lowerLimit, rawRow));
		map.set(task.task_id, clamped);
	}
	return map;
};

export const getConversationRow = (
	entry: VizConversation,
	axisRow: number,
	height: number,
	index: number
): number => {
	if (entry.lane === 'system') {
		return Math.max(1, 1 + (index % 2));
	}
	const base = Math.max(axisRow + 4, height - 4);
	return Math.min(height - 2, base + (entry.role === 'user' ? 0 : 1));
};

export const getConversationTag = (entry: VizConversation): string => {
	if (entry.kind === 'delegate_notice') {
		return 'DG';
	}
	return entry.role === 'user' ? 'U' : entry.role === 'mr1' ? 'M' : 'S';
};

export const buildMergedTranscript = (
	snapshot: VizSnapshot,
	extra: VizConversation[]
): VizConversation[] =>
	[...snapshot.conversation, ...extra]
		.sort((left, right) => left.timestamp.localeCompare(right.timestamp))
		.slice(-8);
