import React, {useEffect, useMemo, useRef, useState} from 'react';
import {
	Box,
	Newline,
	Text,
	render,
	useApp,
	useInput,
	useStdin,
	useStdout,
	useWindowSize
} from 'ink';
import process from 'node:process';
import {BridgeClient, type BridgeMessage} from './bridge.js';
import {loadSnapshot, type VizConversation, type VizSnapshot, type VizTask} from './snapshot.js';
import {
	ROYGBIV,
	buildMergedTranscript,
	buildTaskRowMap,
	colorForPath,
	dimColor,
	getCameraCenterMs,
	getConversationRow,
	getConversationTag,
	getTaskHeadTimeMs,
	isTaskLive,
	type CameraState
} from './timeline.js';

type Cell = {
	char: string;
	fg?: string;
	bg?: string;
	dim?: boolean;
};

const projectRoot = process.env.MR1_PROJECT_ROOT ?? process.cwd();
const pythonBin = process.env.MR1_PYTHON_BIN ?? 'python';
const once = process.argv.includes('--once');

const createCanvas = (width: number, height: number): Cell[][] =>
	Array.from({length: height}, () =>
		Array.from({length: width}, () => ({
			char: ' '
		}))
	);

const setCell = (
	canvas: Cell[][],
	row: number,
	col: number,
	char: string,
	fg?: string,
	bg?: string,
	dim = false
) => {
	if (row < 0 || row >= canvas.length || col < 0 || col >= canvas[0].length) {
		return;
	}
	canvas[row]![col] = {char, fg, bg, dim};
};

const drawText = (
	canvas: Cell[][],
	row: number,
	col: number,
	text: string,
	fg?: string,
	bg?: string,
	dim = false
) => {
	for (let index = 0; index < text.length; index += 1) {
		setCell(canvas, row, col + index, text[index]!, fg, bg, dim);
	}
};

const drawHorizontal = (
	canvas: Cell[][],
	row: number,
	start: number,
	end: number,
	char: string,
	fg?: string,
	bg?: string,
	dim = false
) => {
	const left = Math.min(start, end);
	const right = Math.max(start, end);
	for (let col = left; col <= right; col += 1) {
		setCell(canvas, row, col, char, fg, bg, dim);
	}
};

const drawVertical = (
	canvas: Cell[][],
	col: number,
	start: number,
	end: number,
	char: string,
	fg?: string,
	bg?: string,
	dim = false
) => {
	const top = Math.min(start, end);
	const bottom = Math.max(start, end);
	for (let row = top; row <= bottom; row += 1) {
		setCell(canvas, row, col, char, fg, bg, dim);
	}
};

const fillRect = (
	canvas: Cell[][],
	top: number,
	left: number,
	height: number,
	width: number,
	bg: string
) => {
	for (let row = top; row < top + height; row += 1) {
		for (let col = left; col < left + width; col += 1) {
			setCell(canvas, row, col, ' ', undefined, bg);
		}
	}
};

const drawSprite = (
	canvas: Cell[][],
	top: number,
	left: number,
	sprite: Array<Array<{char: string; fg?: string; bg?: string; dim?: boolean}>>
) => {
	sprite.forEach((row, rowIndex) => {
		row.forEach((cell, colIndex) => {
			if (cell.char !== ' ') {
				setCell(
					canvas,
					top + rowIndex,
					left + colIndex,
					cell.char,
					cell.fg,
					cell.bg,
					cell.dim ?? false
				);
			}
		});
	});
};

const drawRuler = (
	canvas: Cell[][],
	row: number,
	width: number,
	centerCol: number,
	label: string
) => {
	if (row < 0 || row >= canvas.length) {
		return;
	}
	drawHorizontal(canvas, row, 2, width - 3, '·', '#2d2d33');
	for (let col = 6; col < width - 6; col += 8) {
		const tick = col === centerCol ? '·' : '┊';
		const color = col === centerCol ? '#2d2d33' : '#414149';
		setCell(canvas, row, col, tick, color);
	}
	drawText(canvas, row, 2, 'PAST', '#66666d');
	drawText(canvas, row, Math.max(2, width - 8), 'FUTURE', '#66666d');
	drawText(canvas, row, Math.max(2, centerCol - Math.floor(label.length / 2)), label, '#ffd1cb');
};

const placeChip = (
	candidates: number[],
	endColumns: number[],
	preferredCol: number,
	chipWidth: number,
	minCol: number,
	maxCol: number
): {row: number; col: number} => {
	for (let index = 0; index < candidates.length; index += 1) {
		const start = Math.max(minCol, preferredCol, endColumns[index] ?? minCol);
		if (start + chipWidth <= maxCol) {
			endColumns[index] = start + chipWidth + 3;
			return {row: candidates[index]!, col: start};
		}
	}

	let bestIndex = 0;
	let bestValue = Number.POSITIVE_INFINITY;
	for (let index = 0; index < endColumns.length; index += 1) {
		const value = endColumns[index] ?? minCol;
		if (value < bestValue) {
			bestValue = value;
			bestIndex = index;
		}
	}

	const fallbackCol = clamp(Math.max(minCol, bestValue), minCol, Math.max(minCol, maxCol - chipWidth));
	endColumns[bestIndex] = fallbackCol + chipWidth + 3;
	return {row: candidates[bestIndex]!, col: fallbackCol};
};

const buildMr1Sprite = () => [
	[
		{char: ' ', bg: '#101012'},
		{char: '▗', fg: '#ff6b61', bg: '#101012'},
		{char: '▄', fg: '#ff6b61', bg: '#101012'},
		{char: '▄', fg: '#ff6b61', bg: '#101012'},
		{char: '▄', fg: '#ff6b61', bg: '#101012'},
		{char: '▖', fg: '#ff958e', bg: '#101012'},
		{char: ' ', bg: '#101012'}
	],
	[
		{char: '▐', fg: '#ff5d54', bg: '#101012'},
		{char: '█', fg: '#ff453a', bg: '#101012'},
		{char: '╲', fg: '#170505', bg: '#ff453a'},
		{char: ' ', bg: '#ff453a'},
		{char: '╱', fg: '#170505', bg: '#ff5d54'},
		{char: '█', fg: '#ff5d54', bg: '#101012'},
		{char: '▌', fg: '#ff958e', bg: '#101012'}
	],
	[
		{char: '▐', fg: '#ff5d54', bg: '#101012'},
		{char: '█', fg: '#ff453a', bg: '#101012'},
		{char: '█', fg: '#ff453a', bg: '#101012'},
		{char: '█', fg: '#ff453a', bg: '#101012'},
		{char: '█', fg: '#ff5d54', bg: '#101012'},
		{char: '█', fg: '#ff8a83', bg: '#101012'},
		{char: '▌', fg: '#ff8a83', bg: '#101012'}
	],
	[
		{char: '▐', fg: '#ef4036', bg: '#101012'},
		{char: '█', fg: '#d7392f', bg: '#101012'},
		{char: '▀', fg: '#ff6b61', bg: '#101012'},
		{char: '▀', fg: '#ff6b61', bg: '#101012'},
		{char: '▀', fg: '#ff6b61', bg: '#101012'},
		{char: '█', fg: '#f55a50', bg: '#101012'},
		{char: '▌', fg: '#ff7d74', bg: '#101012'}
	],
	[
		{char: ' ', bg: '#101012'},
		{char: '▝', fg: '#ba2f27', bg: '#101012'},
		{char: '▀', fg: '#ba2f27', bg: '#101012'},
		{char: '▀', fg: '#ba2f27', bg: '#101012'},
		{char: '▀', fg: '#c6342b', bg: '#101012'},
		{char: '▘', fg: '#c6342b', bg: '#101012'},
		{char: ' ', bg: '#101012'}
	]
];

const buildAgentSprite = (color: string, dead: boolean) => {
	const main = dead ? dimColor(color) : color;
	const shade = dead ? dimColor(main) : dimColor(color);
	return [
		[
			{char: '▗', fg: main, bg: '#101012', dim: dead},
			{char: '▄', fg: main, bg: '#101012', dim: dead},
			{char: '▖', fg: shade, bg: '#101012', dim: dead}
		],
		[
			{char: '█', fg: main, bg: '#101012', dim: dead},
			{char: '╲', fg: '#161618', bg: main, dim: dead},
			{char: '█', fg: shade, bg: '#101012', dim: dead}
		],
		[
			{char: '▝', fg: shade, bg: '#101012', dim: dead},
			{char: '▀', fg: shade, bg: '#101012', dim: dead},
			{char: '▘', fg: shade, bg: '#101012', dim: dead}
		]
	];
};

const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value));

const formatRelativeTime = (iso: string | null | undefined): string => {
	if (!iso) {
		return 'n/a';
	}
	const deltaSeconds = Math.max(0, Math.floor((Date.now() - new Date(iso).getTime()) / 1000));
	if (deltaSeconds < 60) {
		return `${deltaSeconds}s`;
	}
	if (deltaSeconds < 3600) {
		return `${Math.floor(deltaSeconds / 60)}m`;
	}
	return `${Math.floor(deltaSeconds / 3600)}h`;
};

const ellipsize = (value: string, limit: number): string =>
	value.length <= limit ? value : `${value.slice(0, limit - 1)}…`;

const normalizeInlineText = (value: string): string => value.replace(/\s+/g, ' ').trim();

const shortSessionId = (value: string | null | undefined): string => {
	if (!value) {
		return 'none';
	}
	return value.length <= 10 ? value : value.slice(0, 10);
};

const compactTranscript = (items: VizConversation[]): VizConversation[] => {
	const sorted = [...items].sort((left, right) => left.timestamp.localeCompare(right.timestamp));
	const compacted: VizConversation[] = [];
	for (const item of sorted) {
		const normalized = normalizeInlineText(item.text);
		const previous = compacted.at(-1);
		if (
			previous &&
			previous.role === item.role &&
			previous.kind === item.kind &&
			normalizeInlineText(previous.text) === normalized
		) {
			compacted[compacted.length - 1] = {...item, text: normalized};
			continue;
		}
		compacted.push({...item, text: normalized});
	}
	return compacted.slice(-8);
};

const CanvasView = ({canvas}: {canvas: Cell[][]}) => (
	<Box flexDirection="column">
		{canvas.map((row, rowIndex) => {
			const segments: Array<{text: string; fg?: string; bg?: string; dim?: boolean}> = [];
			let current = row[0] ?? {char: ' '};
			let text = current.char;
			for (let index = 1; index < row.length; index += 1) {
				const cell = row[index]!;
				if (cell.fg === current.fg && cell.bg === current.bg && cell.dim === current.dim) {
					text += cell.char;
				} else {
					segments.push({text, fg: current.fg, bg: current.bg, dim: current.dim});
					current = cell;
					text = cell.char;
				}
			}
			segments.push({text, fg: current.fg, bg: current.bg, dim: current.dim});
			return (
				<Box key={`row-${rowIndex}`}>
					{segments.map((segment, segmentIndex) => (
						<Text
							key={`seg-${segmentIndex}`}
							color={segment.fg}
							backgroundColor={segment.bg}
							dimColor={segment.dim}
						>
							{segment.text}
						</Text>
					))}
				</Box>
			);
		})}
	</Box>
);

const renderTimelineCanvas = (
	snapshot: VizSnapshot,
	camera: CameraState,
	width: number,
	height: number
): Cell[][] => {
	const canvas = createCanvas(width, height);
	const nowMs = Date.now();
	const axisRow = Math.floor(height / 2);
	const lowerConversationRow = Math.min(height - 4, axisRow + 7);
	const taskRowMap = buildTaskRowMap(snapshot.tasks, axisRow, height);
	const mr1Col = Math.round(width / 2);
	const sessionStartMs = snapshot.session.started_at
		? new Date(snapshot.session.started_at).getTime()
		: nowMs;
	const centerMs = getCameraCenterMs(nowMs, camera);
	const conversationRows = [
		Math.min(height - 3, lowerConversationRow + 2),
		Math.min(height - 3, lowerConversationRow + 4),
		Math.min(height - 3, lowerConversationRow + 6),
	].filter((row, index, items) => items.indexOf(row) === index);
	const systemRows = [3, 5, 7].filter(row => row < axisRow - 1);
	const conversationEnds = conversationRows.map(() => 16);
	const systemEnds = systemRows.map(() => 16);

	fillRect(canvas, 0, 0, height, width, '#101012');
	drawRuler(canvas, 1, width, mr1Col, ' NOW ');
	drawHorizontal(canvas, axisRow, 0, width - 1, '─', '#6b5b63');
	drawHorizontal(canvas, axisRow - 6, 12, width - 4, '·', '#25252b');
	drawHorizontal(canvas, lowerConversationRow, 12, width - 4, '·', '#25313d');
	drawVertical(canvas, mr1Col, Math.max(2, axisRow - 6), Math.min(height - 2, lowerConversationRow + 3), '│', '#39393f');
	drawText(canvas, Math.max(2, axisRow - 7), 2, 'SYSTEM', '#7b6b96');
	drawText(canvas, Math.max(2, lowerConversationRow - 1), 2, 'CHAT', '#6aa8f6');
	drawText(canvas, Math.max(2, axisRow - 7), Math.max(2, width - 24), camera.follow ? 'camera locked' : 'camera free', '#66666d');

	const mr1TrailStart = clamp(
		Math.round((sessionStartMs - centerMs) / (camera.secondsPerCell * 1000) + width / 2),
		0,
		width - 1
	);
	drawHorizontal(canvas, axisRow, mr1TrailStart, mr1Col, '═', '#943231', undefined, true);
	drawHorizontal(canvas, axisRow, mr1Col, width - 1, '·', '#4a4a50');
	drawHorizontal(canvas, axisRow - 1, mr1Col - 5, mr1Col + 5, '─', '#2c2c31');
	drawHorizontal(canvas, axisRow + 1, mr1Col - 5, mr1Col + 5, '─', '#2c2c31');
	drawVertical(canvas, mr1Col - 5, axisRow - 1, axisRow + 1, '│', '#2c2c31');
	drawVertical(canvas, mr1Col + 5, axisRow - 1, axisRow + 1, '│', '#2c2c31');
	setCell(canvas, axisRow - 1, mr1Col - 5, '╭', '#2c2c31');
	setCell(canvas, axisRow - 1, mr1Col + 5, '╮', '#2c2c31');
	setCell(canvas, axisRow + 1, mr1Col - 5, '╰', '#2c2c31');
	setCell(canvas, axisRow + 1, mr1Col + 5, '╯', '#2c2c31');

	const sortedTasks = [...snapshot.tasks].sort((left, right) => {
		const leftPath = left.path ?? [];
		const rightPath = right.path ?? [];
		return leftPath.length - rightPath.length || left.task_id.localeCompare(right.task_id);
	});

	for (const task of sortedTasks) {
		const row = taskRowMap.get(task.task_id);
		if (row === undefined || !task.started_at) {
			continue;
		}
		const startMs = new Date(task.started_at).getTime();
		const headMs = getTaskHeadTimeMs(task, nowMs);
		const startCol = clamp(
			Math.round((startMs - centerMs) / (camera.secondsPerCell * 1000) + width / 2),
			0,
			width - 1
		);
		const headCol = clamp(
			Math.round((headMs - centerMs) / (camera.secondsPerCell * 1000) + width / 2),
			0,
			width - 1
		);
		const palette = colorForPath(task.path ?? []);
		const live = isTaskLive(task);
		const color = live ? palette.color : dimColor(palette.color);
		const dim = !live;

		drawHorizontal(canvas, row, startCol, headCol, live ? '━' : '─', color, undefined, dim);

		if (live) {
			const parentRow =
				task.parent_task_id && task.parent_task_id !== 'mr1'
					? (taskRowMap.get(task.parent_task_id) ?? axisRow)
					: axisRow;
			drawVertical(canvas, headCol, parentRow, row, '│', color);
		}

		drawSprite(canvas, row - 1, headCol - 1, buildAgentSprite(color, !live));
		drawText(
			canvas,
			row + 2,
			clamp(headCol + 3, 0, Math.max(0, width - 18)),
			ellipsize(task.description || task.task_id, 18),
			color,
			undefined,
			dim
		);
	}

	if (sortedTasks.length === 0) {
		drawText(canvas, axisRow - 3, Math.max(2, mr1Col - 11), 'no live branches', '#4a4a50');
		drawText(canvas, axisRow + 3, Math.max(2, mr1Col - 15), 'ask MR1 to delegate work', '#3f5566');
	}

	const systemEvents = snapshot.events.filter(event => event.lane === 'system').slice(0, 4);
	systemEvents.forEach((event, index) => {
		const eventMs = new Date(event.timestamp).getTime();
		const preferredCol = clamp(
			Math.round((eventMs - centerMs) / (camera.secondsPerCell * 1000) + width / 2),
			14,
			width - 20
		);
		const label = ` ${ellipsize(event.summary, 12)} `;
		const chip = placeChip(systemRows, systemEnds, preferredCol, label.length + 2, 14, width - 3);
		drawVertical(canvas, chip.col + 1, axisRow - 1, chip.row + 1, '│', '#5a456f', undefined, true);
		drawText(canvas, chip.row, chip.col, '▐', '#bf5af2');
		drawText(canvas, chip.row, chip.col + 1, label, '#dcc0ff', '#23182c');
		drawText(canvas, chip.row, chip.col + label.length + 1, '▌', '#8b46bb');
	});

	const convo = snapshot.conversation.slice(-6);
	convo.forEach((entry, index) => {
		const timeMs = new Date(entry.timestamp).getTime();
		const preferredCol = clamp(
			Math.round((timeMs - centerMs) / (camera.secondsPerCell * 1000) + width / 2),
			14,
			width - 20
		);
		const color = entry.role === 'user' ? '#66c9ff' : entry.role === 'mr1' ? '#ffcaab' : '#a0a0a7';
		const bg = entry.role === 'user' ? '#112235' : entry.role === 'mr1' ? '#2a1c16' : '#18181b';
		const label = ` ${getConversationTag(entry)} ${ellipsize(entry.text.replace(/\s+/g, ' '), 15)} `;
		const chip = placeChip(conversationRows, conversationEnds, preferredCol, label.length + 2, 14, width - 3);
		drawVertical(canvas, chip.col + 1, lowerConversationRow + 1, chip.row - 1, '│', dimColor(color), undefined, true);
		drawText(canvas, chip.row, chip.col, '◉', color);
		drawText(
			canvas,
			chip.row,
			chip.col + 2,
			label,
			color,
			bg
		);
	});

	drawSprite(canvas, axisRow - 2, mr1Col - 3, buildMr1Sprite());
	drawText(canvas, axisRow - 3, clamp(mr1Col + 6, 0, width - 10), 'MR1', '#ff9b92');
	drawText(
		canvas,
		axisRow - 8,
		clamp(mr1Col - 7, 0, width - 16),
		camera.follow ? ' FOLLOW LOCK ' : ' FREE PAN ',
		camera.follow ? '#99f7a5' : '#ffd60a',
		'#18181b'
	);
	return canvas;
};

const TimelineFooter = ({
	camera,
	busy,
	buffer
}: {
	camera: CameraState;
	busy: boolean;
	buffer: string;
}) => (
	<Box flexDirection="column" marginTop={1}>
		<Box justifyContent="space-between">
			<Text color="#8e8e93">
				{`follow ${camera.follow ? 'on' : 'off'}   pan ${camera.panSeconds.toFixed(0)}s   zoom ${camera.secondsPerCell.toFixed(1)}s/cell`}
			</Text>
			<Text color={busy ? '#ffd60a' : '#8e8e93'}>
				{busy ? 'MR1 is thinking…' : 'timeline tree'}
			</Text>
		</Box>
		<Text color="#6f6f76">
			{buffer.length === 0
				? 'controls: arrows or h/l pan, -/= zoom, f recenter on MR1, q quit'
				: busy
					? 'controls: wait for the current turn to finish'
					: 'controls: enter send, backspace delete'}
		</Text>
	</Box>
);

const Transcript = ({
	items,
	height,
	textLimit
}: {
	items: VizConversation[];
	height: number;
	textLimit: number;
}) => (
	<Box
		borderStyle="round"
		borderColor="#403d46"
		flexDirection="column"
		paddingX={1}
		height={height}
	>
		<Text bold color="#f5f5f7">
			Conversation
		</Text>
		{items.length === 0 ? (
			<Text color="#8e8e93">No conversation yet.</Text>
		) : (
			items.map((item, index) => {
				const label =
					item.role === 'user' ? 'YOU' : item.role === 'mr1' ? 'MR1' : item.kind.toUpperCase().slice(0, 5);
				const color =
					item.role === 'user' ? '#66c9ff' : item.role === 'mr1' ? '#ffd2b7' : '#a0a0a7';
				const line = `${label.padEnd(5)} ${item.timestamp.slice(11, 19)} ${ellipsize(normalizeInlineText(item.text), textLimit)}`;
				return (
					<Box key={`${item.timestamp}-${index}`}>
						<Text color={color}>{line}</Text>
					</Box>
				);
			})
		)}
	</Box>
);

const Composer = ({
	buffer,
	busy,
	bridgeReady,
	fatalError
}: {
	buffer: string;
	busy: boolean;
	bridgeReady: boolean;
	fatalError: string | null;
}) => (
	<Box
		borderStyle="round"
		borderColor={fatalError ? '#8e3b46' : busy ? '#6f5a1d' : '#403d46'}
		flexDirection="column"
		paddingX={1}
	>
		<Text bold color="#f5f5f7">
			Prompt
		</Text>
		{fatalError ? (
			<Text color="#ffb4be">{fatalError}</Text>
		) : (
			<Text color={bridgeReady ? '#f5f5f7' : '#8e8e93'}>
				{busy ? '[waiting] ' : 'you > '}
				{buffer || (bridgeReady ? '' : 'starting bridge...')}
			</Text>
		)}
	</Box>
);

const StatusBadge = ({
	label,
	color,
	backgroundColor
}: {
	label: string;
	color: string;
	backgroundColor: string;
}) => (
	<Text color={color} backgroundColor={backgroundColor}>
		{` ${label} `}
	</Text>
);

const SidePanel = ({
	title,
	children,
	height
}: {
	title: string;
	children: React.ReactNode;
	height?: number;
}) => (
	<Box
		borderStyle="round"
		borderColor="#403d46"
		flexDirection="column"
		paddingX={1}
		height={height}
	>
		<Text bold color="#f5f5f7">
			{title}
		</Text>
		{children}
	</Box>
);

const InputController = ({
	buffer,
	busy,
	bridgeReady,
	fatalError,
	setBusy,
	setBuffer,
	setCamera,
	bridge,
	onExit
}: {
	buffer: string;
	busy: boolean;
	bridgeReady: boolean;
	fatalError: string | null;
	setBusy: React.Dispatch<React.SetStateAction<boolean>>;
	setBuffer: React.Dispatch<React.SetStateAction<string>>;
	setCamera: React.Dispatch<React.SetStateAction<CameraState>>;
	bridge: BridgeClient | null;
	onExit: () => void;
}) => {
	useInput((input, key) => {
		if ((input === 'q' && buffer.length === 0) || key.escape || (key.ctrl && input === 'c')) {
			bridge?.close();
			onExit();
			return;
		}

		if (key.leftArrow) {
			setCamera(previous => ({...previous, follow: false, panSeconds: previous.panSeconds - previous.secondsPerCell * 4}));
			return;
		}
		if (key.rightArrow) {
			setCamera(previous => ({...previous, follow: false, panSeconds: previous.panSeconds + previous.secondsPerCell * 4}));
			return;
		}
		if (buffer.length === 0 && input === 'h') {
			setCamera(previous => ({...previous, follow: false, panSeconds: previous.panSeconds - previous.secondsPerCell * 4}));
			return;
		}
		if (buffer.length === 0 && input === 'l') {
			setCamera(previous => ({...previous, follow: false, panSeconds: previous.panSeconds + previous.secondsPerCell * 4}));
			return;
		}

		if (buffer.length === 0 && input === 'f') {
			setCamera(previous => ({...previous, follow: true, panSeconds: 0}));
			return;
		}
		if (buffer.length === 0 && input === '-') {
			setCamera(previous => ({...previous, follow: false, secondsPerCell: Math.min(32, previous.secondsPerCell * 2)}));
			return;
		}
		if (buffer.length === 0 && input === '=') {
			setCamera(previous => ({...previous, follow: false, secondsPerCell: Math.max(0.5, previous.secondsPerCell / 2)}));
			return;
		}

		if (key.return) {
			const text = buffer.trim();
			if (!busy && bridgeReady && !fatalError && text) {
				bridge?.sendInput(text);
				setBusy(true);
				setBuffer('');
			}
			return;
		}

		if (key.backspace || key.delete) {
			setBuffer(previous => previous.slice(0, -1));
			return;
		}

		if (!key.ctrl && !key.meta && input) {
			setBuffer(previous => previous + input);
		}
	});

	return null;
};

const App = () => {
	const {exit} = useApp();
	const {isRawModeSupported} = useStdin();
	const {stdout} = useStdout();
	const {columns, rows} = useWindowSize();
	const bridgeRef = useRef<BridgeClient | null>(null);
	const [snapshot, setSnapshot] = useState<VizSnapshot | null>(null);
	const [bridgeReady, setBridgeReady] = useState(false);
	const [busy, setBusy] = useState(false);
	const [buffer, setBuffer] = useState('');
	const [camera, setCamera] = useState<CameraState>({
		follow: true,
		panSeconds: 0,
		secondsPerCell: 2
	});
	const [extras, setExtras] = useState<VizConversation[]>([]);
	const [statusMessage, setStatusMessage] = useState<string>('starting MR1 bridge...');
	const [fatalError, setFatalError] = useState<string | null>(null);
	const [lastError, setLastError] = useState<string | null>(null);

	useEffect(() => {
		if (once) {
			void loadSnapshot(projectRoot, pythonBin)
				.then(loaded => {
					setSnapshot(loaded);
					setBridgeReady(true);
					setStatusMessage('snapshot-only render');
					setTimeout(() => exit(), 50);
				})
				.catch(error => {
					const message = error instanceof Error ? error.message : String(error);
					setFatalError(message);
					setStatusMessage('snapshot load failed');
				});
			return;
		}

		const handleMessage = (message: BridgeMessage) => {
			if (message.type === 'ready') {
				setBridgeReady(true);
				setFatalError(null);
				setStatusMessage(`session ${message.session_id}`);
				return;
			}
			if (message.type === 'snapshot') {
				setSnapshot(message.snapshot);
				return;
			}
			if (message.type === 'response') {
				setBusy(false);
				return;
			}
			if (message.type === 'command_result') {
				setBusy(false);
				setExtras(previous => [
					...previous,
					{
						timestamp: new Date().toISOString(),
						role: 'system',
						text: message.output,
						kind: 'command',
						lane: 'conversation'
					}
				]);
				return;
			}
			if (message.type === 'error') {
				setBusy(false);
				setLastError(message.message);
				setStatusMessage(message.message);
				if (message.fatal) {
					setFatalError(message.message);
					setBridgeReady(false);
				}
				setExtras(previous => [
					...previous,
					{
						timestamp: new Date().toISOString(),
						role: 'system',
						text: message.message,
						kind: 'error',
						lane: 'system'
					}
				]);
				return;
			}
			if (message.type === 'event' && message.event.summary) {
				setStatusMessage(message.event.summary);
				return;
			}
			if (message.type === 'shutdown') {
				setBusy(false);
			}
		};

		const bridge = new BridgeClient(projectRoot, pythonBin, handleMessage);
		bridgeRef.current = bridge;
		bridge.start();

		return () => {
			bridge.close();
		};
	}, []);

	const transcriptItems = useMemo(
		() => compactTranscript(snapshot ? buildMergedTranscript(snapshot, extras) : extras.slice(-8)),
		[snapshot, extras]
	);
	const interactive = stdout.isTTY && isRawModeSupported && !once;

	if (!snapshot) {
		return (
			<Box flexDirection="column">
				<Text color="#f5f5f7" bold>
					MR1 Timeline UI
				</Text>
				<Text color="#8e8e93">{statusMessage}</Text>
			</Box>
		);
	}

	const timelineHeight = Math.max(16, Math.min(26, rows - 12));
	const canvasWidth = Math.max(60, columns - 2);
	const timelineCanvas = renderTimelineCanvas(snapshot, camera, canvasWidth, timelineHeight);
	const stackedLowerPanels = columns < 132;
	const modeLabel = once ? 'SNAPSHOT' : 'LIVE';
	const readyLabel = fatalError ? 'OFFLINE' : bridgeReady ? 'READY' : 'BOOTING';
	const latestConversation = transcriptItems.at(-1);
	const displayedTranscriptItems = transcriptItems.slice(stackedLowerPanels ? -6 : -5);

	return (
		<Box flexDirection="column">
			{interactive ? (
				<InputController
					buffer={buffer}
					busy={busy}
					bridgeReady={bridgeReady}
					fatalError={fatalError}
					setBusy={setBusy}
					setBuffer={setBuffer}
					setCamera={setCamera}
					bridge={bridgeRef.current}
					onExit={() => exit()}
				/>
			) : null}
			<Box justifyContent="space-between">
				<Box flexDirection="column">
					<Box>
						<Text bold color="#f5f5f7">
							MR1 Timeline Tree UI
						</Text>
						<Text color="#8e8e93">{`  ${statusMessage}`}</Text>
					</Box>
					<Box marginTop={1}>
						<StatusBadge
							label={modeLabel}
							color={once ? '#d7c4ff' : '#99f7a5'}
							backgroundColor={once ? '#251a31' : '#18251a'}
						/>
						<Text> </Text>
						<StatusBadge
							label={readyLabel}
							color={fatalError ? '#ffb4be' : bridgeReady ? '#b9fbc0' : '#ffe38a'}
							backgroundColor={fatalError ? '#34171d' : bridgeReady ? '#18301d' : '#362b12'}
						/>
						{busy ? (
							<>
								<Text> </Text>
								<StatusBadge label="BUSY" color="#ffe38a" backgroundColor="#362b12" />
							</>
						) : null}
					</Box>
				</Box>
				<Box flexDirection="column" alignItems="flex-end">
					<Text color="#8e8e93">{`session ${shortSessionId(snapshot.session.session_id)}`}</Text>
					<Text color="#6f6f76">{`running ${snapshot.summary.running_count}   tasks ${snapshot.summary.task_count}`}</Text>
				</Box>
			</Box>
			<Box marginTop={1} borderStyle="round" borderColor="#403d46" paddingX={1} flexDirection="column">
				<CanvasView canvas={timelineCanvas} />
			</Box>
			<TimelineFooter camera={camera} busy={busy} buffer={buffer} />
			<Newline />
			{stackedLowerPanels ? (
				<Box flexDirection="column">
					<Transcript items={displayedTranscriptItems} height={8} textLimit={56} />
					<Box marginTop={1}>
						<SidePanel title="Status">
							<Text color={fatalError ? '#ffb4be' : '#c4c4ca'}>
								{fatalError ?? lastError ?? statusMessage}
							</Text>
							<Text color="#6f6f76">
								{latestConversation
									? `latest: ${latestConversation.role} at ${latestConversation.timestamp.slice(11, 19)}`
									: 'latest: no turns yet'}
							</Text>
						</SidePanel>
					</Box>
					<Box marginTop={1}>
						<SidePanel title="Controls">
							<Text color="#c4c4ca">enter send a prompt</Text>
							<Text color="#c4c4ca">f relock camera on MR1</Text>
							<Text color="#c4c4ca">h/l or arrows scrub time</Text>
							<Text color="#c4c4ca">-/= zoom the timeline</Text>
							<Text color="#c4c4ca">q quit the UI</Text>
						</SidePanel>
					</Box>
				</Box>
			) : (
				<Box flexDirection="row">
					<Box flexGrow={1}>
						<Transcript items={displayedTranscriptItems} height={11} textLimit={28} />
					</Box>
					<Box width={42} marginLeft={1} flexDirection="column">
						<SidePanel title="Status" height={6}>
							<Text color={fatalError ? '#ffb4be' : '#c4c4ca'}>
								{fatalError ?? lastError ?? statusMessage}
							</Text>
							<Text color="#6f6f76">
								{latestConversation
									? `latest: ${latestConversation.role} at ${latestConversation.timestamp.slice(11, 19)}`
									: 'latest: no turns yet'}
							</Text>
						</SidePanel>
						<Box marginTop={1} />
						<SidePanel title="Controls" height={7}>
							<Text color="#c4c4ca">enter send a prompt</Text>
							<Text color="#c4c4ca">f relock camera on MR1</Text>
							<Text color="#c4c4ca">h/l or arrows scrub time</Text>
							<Text color="#c4c4ca">-/= zoom the timeline</Text>
							<Text color="#c4c4ca">q quit the UI</Text>
						</SidePanel>
					</Box>
				</Box>
			)}
			<Box marginTop={1}>
				<Composer buffer={buffer} busy={busy} bridgeReady={bridgeReady} fatalError={fatalError} />
			</Box>
			<Box marginTop={1} justifyContent="space-between">
				<Box>
					<Text color="#ff453a">R</Text>
					<Text color="#ff9f0a">O</Text>
					<Text color="#ffd60a">Y</Text>
					<Text color="#30d158">G</Text>
					<Text color="#0a84ff">B</Text>
					<Text color="#5e5ce6">I</Text>
					<Text color="#bf5af2">V</Text>
					<Text color="#8e8e93"> branch path color</Text>
				</Box>
				<Text color="#8e8e93">dead agents freeze, dim, and detach from the live tree</Text>
			</Box>
			{stdout.isTTY ? null : (
				<Text color="#8e8e93">{`snapshot ${formatRelativeTime(snapshot.generated_at)} ago`}</Text>
			)}
		</Box>
	);
};

render(<App />, {exitOnCtrlC: true, interactive: !once && process.stdout.isTTY});
