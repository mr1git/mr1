/** @jsxImportSource @termuijs/jsx */

import process from 'node:process';
import {render, useEffect, useInput, useInterval, useMemo, useState} from '@termuijs/jsx';
import type {Color} from '@termuijs/core';
import type {BridgeMessage} from '../viz/bridge.js';
import {BridgeClient} from '../viz/bridge.js';
import {loadSnapshot, type VizConversation, type VizSnapshot} from '../viz/snapshot.js';
import type {CameraState} from '../viz/timeline.js';
import {renderTimeline, type StyledSegment} from './render.js';

const projectRoot = process.env.MR1_PROJECT_ROOT ?? process.cwd();
const pythonBin = process.env.MR1_PYTHON_BIN ?? process.env.PYTHON ?? 'python3';
const onceMode = process.argv.includes('--once');
const fullscreenMode = process.env.MR1_TERMUI_FULLSCREEN === '1';
let activeBridge: BridgeClient | null = null;

const BOX_BORDER = '#5f6470';
const MUTED = '#a4aaba';
const ACCENT = '#ff6f72';
const OK = '#7ef1a9';
const WARN = '#ffd479';
const ERROR = '#ff8d95';

type AppState = {
	snapshot: VizSnapshot | null;
	extraConversation: VizConversation[];
	sessionId: string | null;
	connected: boolean;
	busy: boolean;
	fatalError: string | null;
	lastError: string | null;
};

type LayoutState = {
	columns: number;
	sidebarWidth: number;
	timelineWidth: number;
};

const hex = (value: string): Color => ({type: 'hex', hex: value});

const conversationRoleColor = (role: string): Color =>
	role === 'user' ? hex('#58d7ff') : role === 'mr1' ? hex('#ffd27a') : hex('#a7b1d8');

const formatClock = (iso: string): string => {
	const date = new Date(iso);
	if (Number.isNaN(date.getTime())) {
		return '--:--:--';
	}
	return date.toLocaleTimeString([], {hour12: false});
};

const clip = (value: string, width: number): string =>
	value.length <= width ? value : `${value.slice(0, Math.max(0, width - 1))}…`;

const measureLayout = (): LayoutState => {
	const columns = process.stdout.columns ?? 120;
	const sidebarWidth = Math.max(24, Math.min(32, Math.floor(columns * 0.24)));
	const timelineWidth = Math.max(48, columns - sidebarWidth - 10);
	return {columns, sidebarWidth, timelineWidth};
};

const SegmentRow = ({segments}: {segments: StyledSegment[]}) => (
	<row gap={0}>
		{segments.map((segment, index) => (
			<text
				key={index}
				color={segment.color}
				dim={segment.dim}
				bold={segment.bold}
			>
				{segment.text}
			</text>
		))}
	</row>
);

const TranscriptPanel = ({entries}: {entries: VizConversation[]}) => (
	<col gap={0}>
		{textHeading('Conversation')}
		{entries.slice(-6).map((entry, index) => (
			<row key={index} gap={1}>
				<text color={conversationRoleColor(entry.role)} bold>
					{entry.role === 'user' ? 'YOU' : entry.role === 'mr1' ? 'MR1' : 'SYS'}
				</text>
				<text color={hex(MUTED)}>{formatClock(entry.timestamp)}</text>
				<text>{clip(entry.text.replace(/\s+/g, ' ').trim(), 58)}</text>
			</row>
		))}
	</col>
);

const textHeading = (title: string) => (
	<text color={hex('#f2f4f8')} bold>
		{title}
	</text>
);

function App() {
	const [camera, setCamera] = useState<CameraState>({
		follow: true,
		panSeconds: 0,
		secondsPerCell: 2
	});
	const [prompt, setPrompt] = useState('');
	const [nowMs, setNowMs] = useState(Date.now());
	const [layout, setLayout] = useState<LayoutState>(measureLayout());
	const [state, setState] = useState<AppState>({
		snapshot: null,
		extraConversation: [],
		sessionId: null,
		connected: onceMode,
		busy: false,
		fatalError: null,
		lastError: null
	});

	useEffect(() => {
		if (onceMode) {
			void loadSnapshot(projectRoot, pythonBin)
				.then(snapshot => {
					setState(current => ({
						...current,
						snapshot,
						connected: true
					}));
				})
				.catch(error => {
					setState(current => ({
						...current,
						fatalError: error instanceof Error ? error.message : String(error)
					}));
				});
			return;
		}

		const bridge = new BridgeClient(projectRoot, pythonBin, (message: BridgeMessage) => {
			setState(current => reduceBridgeMessage(current, message));
		}, {
			MR1_UI_SNAPSHOT_INTERVAL_S: '0.9'
		});
		activeBridge = bridge;
		bridge.start();
		return () => {
			activeBridge = null;
			bridge.close();
		};
	}, []);

	useInterval(() => {
		setNowMs(Date.now());
		const next = measureLayout();
		setLayout(current =>
			current.columns === next.columns &&
			current.sidebarWidth === next.sidebarWidth &&
			current.timelineWidth === next.timelineWidth
				? current
				: next
		);
	}, 600);

	useInput((key, event) => {
		if (event.ctrl && key === 'c') {
			process.exit(0);
		}

		if (!prompt && key === 'q') {
			process.exit(0);
		}

		if (key === 'left' || key === 'h') {
			setCamera(current => ({...current, follow: false, panSeconds: current.panSeconds - current.secondsPerCell * 2}));
			return;
		}
		if (key === 'right' || key === 'l') {
			setCamera(current => ({...current, follow: false, panSeconds: current.panSeconds + current.secondsPerCell * 2}));
			return;
		}
		if (key === '-' || key === '_') {
			setCamera(current => ({...current, secondsPerCell: Math.min(12, Number((current.secondsPerCell * 1.35).toFixed(2)))}));
			return;
		}
		if (key === '=' || key === '+') {
			setCamera(current => ({...current, secondsPerCell: Math.max(0.35, Number((current.secondsPerCell / 1.35).toFixed(2)))}));
			return;
		}
		if (key === 'f') {
			setCamera(current => ({...current, follow: true, panSeconds: 0}));
			return;
		}
		if (key === 'backspace') {
			setPrompt(current => current.slice(0, -1));
			return;
		}
		if (key === 'return' || key === 'enter') {
			submitPrompt(prompt, setState, setPrompt);
			return;
		}
		if (onceMode) {
			return;
		}
		if (!event.ctrl) {
			if (key === 'space') {
				setPrompt(current => `${current} `);
				return;
			}
			if (key.length === 1) {
				setPrompt(current => `${current}${key}`);
			}
		}
	});

	const timeline = useMemo(() => {
		if (!state.snapshot) {
			return null;
		}
		return renderTimeline(
			state.snapshot,
			camera,
			layout.timelineWidth,
			nowMs,
			state.extraConversation
		);
	}, [state.snapshot, state.extraConversation, camera, nowMs, layout.timelineWidth]);

	const transcript = timeline?.transcript ?? state.extraConversation;
	const runningCount = state.snapshot?.summary.running_count ?? 0;
	const statusTone = state.fatalError ? hex(ERROR) : state.busy ? hex(WARN) : hex(OK);
	const statusText = state.fatalError
		? 'bridge offline'
		: state.busy
			? 'mr1 thinking'
			: state.connected
				? 'connected'
				: 'starting';

	return (
		<col padding={1} gap={1}>
			<row>
				<text color={hex(ACCENT)} bold>
					MR1 / TermUI prototype
				</text>
				<spacer />
				<text color={statusTone} bold>
					{statusText.toUpperCase()}
				</text>
			</row>

			<row gap={1}>
				<box border="rounded" borderColor={hex(BOX_BORDER)} padding={1} flexGrow={1}>
					<col gap={0}>
						<row>
							<text color={hex('#f4f5f8')} bold>
								live timeline
							</text>
							<spacer />
							<text color={hex(MUTED)}>
								{camera.follow ? 'follow lock' : `scrub ${camera.panSeconds.toFixed(1)}s`}
							</text>
						</row>
						<divider color={hex('#3d4354')} />
						{timeline ? (
							timeline.rows.map((segments, index) => <SegmentRow key={index} segments={segments} />)
						) : (
							<col gap={1}>
								<text color={hex(MUTED)}>waiting for MR1 timeline data…</text>
								{state.fatalError ? (
									<text color={hex(ERROR)}>{state.fatalError}</text>
								) : null}
							</col>
						)}
					</col>
				</box>

				<box border="rounded" borderColor={hex(BOX_BORDER)} padding={1} width={layout.sidebarWidth}>
					<col gap={1}>
						{textHeading('Status')}
						<text color={hex(MUTED)}>session {state.sessionId ?? 'pending'}</text>
						<text>{`running workers ${runningCount}`}</text>
						<text>{`zoom ${camera.secondsPerCell.toFixed(2)}s/cell`}</text>
						{textHeading('Keys')}
						<text>enter send</text>
						<text>h/l or arrows scrub</text>
						<text>- / = zoom</text>
						<text>f relock</text>
						<text>q quit when prompt empty</text>
						{state.lastError ? (
							<>
								{textHeading('Latest')}
								<text color={hex(ERROR)}>{clip(state.lastError, 24)}</text>
							</>
						) : null}
					</col>
				</box>
			</row>

			<row gap={1}>
				<box border="rounded" borderColor={hex(BOX_BORDER)} padding={1} flexGrow={1}>
					<TranscriptPanel entries={transcript} />
				</box>

				<box border="rounded" borderColor={hex(BOX_BORDER)} padding={1} width={layout.sidebarWidth}>
					<col gap={1}>
						{textHeading('Prompt')}
						<text color={hex('#f4f5f8')} bold>
							you {'>'} {prompt || ' '}
						</text>
						<text color={hex(MUTED)} dim>
							{onceMode
								? 'snapshot mode: prompt disabled'
								: state.busy
									? 'waiting for MR1 response…'
									: 'type normally; q exits if prompt is empty'}
						</text>
					</col>
				</box>
			</row>
		</col>
	);
}

const reduceBridgeMessage = (state: AppState, message: BridgeMessage): AppState => {
	switch (message.type) {
		case 'ready':
			return {
				...state,
				sessionId: message.session_id,
				connected: true,
				fatalError: null
			};
		case 'snapshot':
			return {
				...state,
				snapshot: message.snapshot,
				connected: true,
				fatalError: null
			};
		case 'response':
			return {
				...state,
				busy: false,
				lastError: null,
				extraConversation: [
					...state.extraConversation,
					{
						timestamp: new Date().toISOString(),
						role: 'mr1',
						text: message.text,
						kind: 'response',
						lane: 'conversation'
					}
				].slice(-8)
			};
		case 'command_result':
			return {
				...state,
				busy: false,
				extraConversation: [
					...state.extraConversation,
					{
						timestamp: new Date().toISOString(),
						role: 'system',
						text: `${message.command}: ${message.output}`,
						kind: 'command_result',
						lane: 'system'
					}
				].slice(-8)
			};
		case 'error':
			return {
				...state,
				busy: false,
				lastError: message.message,
				fatalError: message.fatal ? message.message : state.fatalError
			};
		case 'shutdown':
			return {
				...state,
				connected: false,
				busy: false
			};
		case 'event':
			return state;
	}
	return state;
};

const submitPrompt = (
	prompt: string,
	setState: (updater: (current: AppState) => AppState) => void,
	setPrompt: (value: string) => void
): void => {
	const value = prompt.trim();
	if (!value || onceMode) {
		return;
	}
	setState(current => ({
		...current,
		busy: true,
		lastError: null,
		extraConversation: [
			...current.extraConversation,
			{
				timestamp: new Date().toISOString(),
				role: 'user',
				text: value,
				kind: 'prompt',
				lane: 'conversation'
			}
		].slice(-8)
	}));
	setPrompt('');
	activeBridge?.sendInput(value);
};

async function main(): Promise<void> {
	if (!process.stdout.isTTY || !process.stdin.isTTY) {
		console.error('TermUI requires an interactive TTY. Use `npm run viz:termui` in a normal terminal.');
		process.exit(1);
	}

	await render(<App />, {
		fullscreen: fullscreenMode
	});
}

void main();
