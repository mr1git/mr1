import test from 'node:test';
import assert from 'node:assert/strict';
import {renderTimeline} from './render.js';
import type {VizSnapshot} from '../viz/snapshot.js';
import type {CameraState} from '../viz/timeline.js';

const camera: CameraState = {
	follow: true,
	panSeconds: 0,
	secondsPerCell: 2
};

const snapshot: VizSnapshot = {
	generated_at: '2026-04-20T12:00:10.000Z',
	session: {
		session_id: 'sess-1',
		started_at: '2026-04-20T12:00:00.000Z',
		status: 'running'
	},
	summary: {
		task_count: 2,
		running_count: 1,
		decision_count: 1
	},
	root: {
		id: 'mr1',
		name: 'MR1',
		status: 'running'
	},
	tasks: [
		{
			task_id: 'child-live',
			parent_task_id: 'mr1',
			agent_type: 'mrn',
			status: 'running',
			description: 'delegated live branch',
			started_at: '2026-04-20T12:00:03.000Z',
			updated_at: '2026-04-20T12:00:09.000Z',
			finished_at: null,
			pid: 111,
			event_count: 3,
			path: [0],
			lane: 'conversation'
		},
		{
			task_id: 'child-dead',
			parent_task_id: 'mr1',
			agent_type: 'kazi',
			status: 'completed',
			description: 'historical branch',
			started_at: '2026-04-20T11:59:59.000Z',
			updated_at: '2026-04-20T12:00:04.000Z',
			finished_at: '2026-04-20T12:00:04.000Z',
			pid: 222,
			event_count: 5,
			path: [1],
			lane: 'system'
		}
	],
	events: [],
	conversation: [
		{
			timestamp: '2026-04-20T12:00:01.000Z',
			role: 'user',
			text: 'please delegate',
			kind: 'prompt',
			lane: 'conversation'
		},
		{
			timestamp: '2026-04-20T12:00:08.000Z',
			role: 'mr1',
			text: 'delegating now',
			kind: 'response',
			lane: 'conversation'
		}
	],
	recent_decisions: []
};

test('renderTimeline keeps MR1 centered', () => {
	const timeline = renderTimeline(snapshot, camera, 80, Date.parse(snapshot.generated_at));
	assert.equal(timeline.centerCol, 40);
	assert.equal(timeline.cells[timeline.axisRow - 1][timeline.centerCol].char, '╱');
});

test('renderTimeline dims dead branches', () => {
	const timeline = renderTimeline(snapshot, camera, 80, Date.parse(snapshot.generated_at));
	const dimCell = timeline.cells.flat().find(cell => cell.char === '▪');
	assert.ok(dimCell);
	assert.equal(dimCell?.dim, true);
});
