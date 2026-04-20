import type {Color} from '@termuijs/core';
import type {VizConversation, VizSnapshot, VizTask} from '../viz/snapshot.js';
import {
	buildMergedTranscript,
	colorForPath,
	dimColor,
	getTaskHeadTimeMs,
	type CameraState,
	worldToColumn
} from '../viz/timeline.js';

export type StyledCell = {
	char: string;
	color: Color;
	dim: boolean;
	bold: boolean;
};

export type StyledSegment = {
	text: string;
	color?: Color;
	dim?: boolean;
	bold?: boolean;
};

export type TimelineRender = {
	cells: StyledCell[][];
	rows: StyledSegment[][];
	axisRow: number;
	centerCol: number;
	transcript: VizConversation[];
};

const NONE: Color = {type: 'none'};
const GRID_HEIGHT = 15;

const SYSTEM_LABEL = '#8ea0ff';
const CHAT_LABEL = '#72e4ff';
const AXIS_COLOR = '#48506a';
const MR1_MAIN = '#ff6868';
const MR1_SHADE = '#cb3c4f';
const MR1_EYES = '#fff1de';
const CHAT_USER = '#5cdcff';
const CHAT_MR1 = '#ffd479';
const CHAT_SYSTEM = '#9aa6cf';

const hex = (value: string): Color => ({type: 'hex', hex: value});

const emptyCell = (): StyledCell => ({
	char: ' ',
	color: NONE,
	dim: false,
	bold: false
});

const createGrid = (width: number, height: number): StyledCell[][] =>
	Array.from({length: height}, () => Array.from({length: width}, () => emptyCell()));

const setCell = (
	grid: StyledCell[][],
	row: number,
	col: number,
	char: string,
	color?: Color,
	options?: {dim?: boolean; bold?: boolean}
): void => {
	if (row < 0 || row >= grid.length) {
		return;
	}
	if (col < 0 || col >= grid[row].length) {
		return;
	}
	grid[row][col] = {
		char,
		color: color ?? NONE,
		dim: options?.dim ?? false,
		bold: options?.bold ?? false
	};
};

const drawText = (
	grid: StyledCell[][],
	row: number,
	col: number,
	text: string,
	color?: Color,
	options?: {dim?: boolean; bold?: boolean}
): void => {
	for (const [offset, char] of Array.from(text).entries()) {
		setCell(grid, row, col + offset, char, color, options);
	}
};

const drawHorizontal = (
	grid: StyledCell[][],
	row: number,
	startCol: number,
	endCol: number,
	char: string,
	color?: Color,
	options?: {dim?: boolean; bold?: boolean}
): void => {
	const left = Math.max(0, Math.min(startCol, endCol));
	const right = Math.min(grid[0]?.length ?? 0, Math.max(startCol, endCol) + 1);
	for (let col = left; col < right; col += 1) {
		setCell(grid, row, col, char, color, options);
	}
};

const drawVertical = (
	grid: StyledCell[][],
	col: number,
	startRow: number,
	endRow: number,
	char: string,
	color?: Color,
	options?: {dim?: boolean; bold?: boolean}
): void => {
	const top = Math.max(0, Math.min(startRow, endRow));
	const bottom = Math.min(grid.length, Math.max(startRow, endRow) + 1);
	for (let row = top; row < bottom; row += 1) {
		setCell(grid, row, col, char, color, options);
	}
};

const drawSprite = (
	grid: StyledCell[][],
	top: number,
	left: number,
	sprite: Array<Array<{char: string; color: Color; dim?: boolean; bold?: boolean}>>
): void => {
	for (const [rowOffset, row] of sprite.entries()) {
		for (const [colOffset, cell] of row.entries()) {
			setCell(grid, top + rowOffset, left + colOffset, cell.char, cell.color, {
				dim: cell.dim,
				bold: cell.bold
			});
		}
	}
};

const clamp = (value: number, min: number, max: number): number =>
	Math.max(min, Math.min(max, value));

const summarize = (text: string, maxWidth: number): string => {
	const clean = text.replace(/\s+/g, ' ').trim();
	if (clean.length <= maxWidth) {
		return clean;
	}
	return `${clean.slice(0, Math.max(0, maxWidth - 1))}…`;
};

const labelColorForRole = (entry: VizConversation): Color => {
	if (entry.role === 'user') {
		return hex(CHAT_USER);
	}
	if (entry.role === 'mr1') {
		return hex(CHAT_MR1);
	}
	return hex(CHAT_SYSTEM);
};

const rowForTask = (task: VizTask, axisRow: number): number => {
	const depth = Math.max(1, task.path?.length ?? 1);
	const direction = task.lane === 'system' ? -1 : 1;
	return axisRow + direction * (1 + depth * 2);
};

const mr1Sprite = (): Array<Array<{char: string; color: Color}>> => [
	[
		{char: '█', color: hex(MR1_MAIN)},
		{char: '█', color: hex(MR1_MAIN)},
		{char: '█', color: hex(MR1_MAIN)},
		{char: '▓', color: hex(MR1_SHADE)}
	],
	[
		{char: '█', color: hex(MR1_MAIN)},
		{char: '╲', color: hex(MR1_EYES)},
		{char: '╱', color: hex(MR1_EYES)},
		{char: '▓', color: hex(MR1_SHADE)}
	],
	[
		{char: '█', color: hex(MR1_MAIN)},
		{char: '█', color: hex(MR1_MAIN)},
		{char: '█', color: hex(MR1_MAIN)},
		{char: '▓', color: hex(MR1_SHADE)}
	],
	[
		{char: '█', color: hex(MR1_MAIN)},
		{char: '█', color: hex(MR1_MAIN)},
		{char: '█', color: hex(MR1_MAIN)},
		{char: '▓', color: hex(MR1_SHADE)}
	]
];

const taskGlyph = (task: VizTask): string => {
	if (task.status === 'failed') {
		return '✕';
	}
	return task.status === 'running' ? '■' : '▪';
};

const compressRow = (row: StyledCell[]): StyledSegment[] => {
	const segments: StyledSegment[] = [];
	let current: StyledSegment | null = null;
	for (const cell of row) {
		const matches =
			current &&
			(current.color ?? NONE).type === cell.color.type &&
			JSON.stringify(current.color ?? NONE) === JSON.stringify(cell.color) &&
			Boolean(current.dim) === cell.dim &&
			Boolean(current.bold) === cell.bold;
		if (!current || !matches) {
			current = {
				text: cell.char,
				color: cell.color.type === 'none' ? undefined : cell.color,
				dim: cell.dim || undefined,
				bold: cell.bold || undefined
			};
			segments.push(current);
		} else {
			current.text += cell.char;
		}
	}
	return segments;
};

export const renderTimeline = (
	snapshot: VizSnapshot,
	camera: CameraState,
	width: number,
	nowMs: number,
	extraConversation: VizConversation[] = []
): TimelineRender => {
	const grid = createGrid(width, GRID_HEIGHT);
	const axisRow = Math.floor(GRID_HEIGHT / 2);
	const centerCol = Math.floor(width / 2);
	const mergedTranscript = buildMergedTranscript(snapshot, extraConversation);
	const tasks = [...snapshot.tasks].sort((left, right) => {
		const depthDelta = (left.path?.length ?? 0) - (right.path?.length ?? 0);
		if (depthDelta !== 0) {
			return depthDelta;
		}
		return left.task_id.localeCompare(right.task_id);
	});
	const taskMap = new Map(tasks.map(task => [task.task_id, task]));

	drawText(grid, 0, 1, 'SYSTEM', hex(SYSTEM_LABEL), {bold: true});
	drawText(grid, GRID_HEIGHT - 2, 1, 'CHAT', hex(CHAT_LABEL), {bold: true});
	drawHorizontal(grid, axisRow, 0, width - 1, '·', hex(AXIS_COLOR), {dim: true});

	const anchorLeft = centerCol - 2;
	drawSprite(grid, axisRow - 2, anchorLeft, mr1Sprite());
	drawText(grid, axisRow + 3, centerCol - 1, 'MR1', hex('#ffc5c5'), {bold: true});

	for (const task of tasks) {
		const palette = colorForPath(task.path ?? []);
		const live = task.status === 'running';
		const baseColor = hex(live ? palette.color : dimColor(palette.color));
		const dim = !live;
		const headTime = getTaskHeadTimeMs(task, nowMs);
		const headCol = clamp(worldToColumn(headTime, nowMs, camera, width), 0, width - 1);
		const startCol = clamp(
			worldToColumn(new Date(task.started_at ?? snapshot.generated_at).getTime(), nowMs, camera, width),
			0,
			width - 1
		);
		const row = clamp(rowForTask(task, axisRow), 1, GRID_HEIGHT - 4);

		if (live) {
			const parent = task.parent_task_id ? taskMap.get(task.parent_task_id) : null;
			const parentRow = parent ? clamp(rowForTask(parent, axisRow), 1, GRID_HEIGHT - 4) : axisRow;
			const parentHeadTime = parent ? getTaskHeadTimeMs(parent, nowMs) : nowMs;
			const parentCol = clamp(worldToColumn(parentHeadTime, nowMs, camera, width), 0, width - 1);
			drawVertical(grid, parentCol, parentRow, row, '│', baseColor, {dim});
			drawHorizontal(grid, row, parentCol, headCol, '─', baseColor, {dim});
		} else {
			drawHorizontal(grid, row, startCol, headCol, '─', baseColor, {dim: true});
		}

		setCell(grid, row, headCol, taskGlyph(task), baseColor, {
			dim,
			bold: live
		});

		const label = summarize(task.description || task.agent_type || task.task_id, 18);
		drawText(grid, row + (task.lane === 'system' ? -1 : 1), clamp(headCol + 2, 0, width - label.length), label, baseColor, {
			dim,
			bold: live
		});
	}

	const chatBaseRow = axisRow + 4;
	for (const [index, entry] of mergedTranscript.slice(-4).entries()) {
		const col = clamp(
			worldToColumn(new Date(entry.timestamp).getTime(), nowMs, camera, width),
			0,
			width - 1
		);
		const row = clamp(chatBaseRow + (index % 2), axisRow + 2, GRID_HEIGHT - 3);
		const prefix = entry.role === 'user' ? 'U' : entry.role === 'mr1' ? 'M' : 'S';
		const chip = `${prefix} ${summarize(entry.text, 20)}`;
		drawText(grid, row, clamp(col, 0, Math.max(0, width - chip.length)), chip, labelColorForRole(entry), {
			bold: entry.role !== 'system'
		});
	}

	return {
		cells: grid,
		rows: grid.map(compressRow),
		axisRow,
		centerCol,
		transcript: mergedTranscript
	};
};
