import test from 'node:test';
import assert from 'node:assert/strict';
import {
	ROYGBIV,
	buildTaskRowMap,
	colorForPath,
	dimColor,
	getCameraCenterMs,
	getTaskHeadTimeMs,
	worldToColumn,
	type CameraState
} from './timeline.js';
import type {VizTask} from './snapshot.js';

const camera: CameraState = {
	follow: true,
	panSeconds: 0,
	secondsPerCell: 2
};

test('colorForPath is stable for the same path', () => {
	assert.deepEqual(colorForPath([0, 1, 2]), colorForPath([0, 1, 2]));
	assert.equal(colorForPath([]).letter, 'R');
	assert.ok(ROYGBIV.some(entry => entry.letter === colorForPath([3]).letter));
});

test('dimColor darkens a branch color', () => {
	assert.notEqual(dimColor('#ff453a'), '#ff453a');
	assert.equal(dimColor('#ffffff'), '#7a7a7a');
});

test('camera center follows now and respects pan', () => {
	const now = 1_700_000_000_000;
	assert.equal(getCameraCenterMs(now, camera), now);
	assert.equal(getCameraCenterMs(now, {...camera, follow: false, panSeconds: -10}), now - 10_000);
});

test('worldToColumn keeps now centered when following', () => {
	const now = 1_700_000_000_000;
	assert.equal(worldToColumn(now, now, camera, 120), 60);
	assert.equal(worldToColumn(now - 4_000, now, camera, 120), 58);
});

test('running tasks use now as head position while dead tasks freeze', () => {
	const now = 1_700_000_000_000;
	const running: VizTask = {
		task_id: 'live',
		parent_task_id: 'mr1',
		agent_type: 'kazi',
		status: 'running',
		description: 'live',
		started_at: '2024-01-01T00:00:00+00:00',
		updated_at: '2024-01-01T00:00:05+00:00',
		finished_at: null,
		pid: 1,
		event_count: 1,
		lane: 'conversation'
	};
	const dead: VizTask = {
		...running,
		task_id: 'dead',
		status: 'completed',
		finished_at: '2024-01-01T00:00:09+00:00'
	};
	assert.equal(getTaskHeadTimeMs(running, now), now);
	assert.equal(getTaskHeadTimeMs(dead, now), new Date('2024-01-01T00:00:09+00:00').getTime());
});

test('task rows stay stable by path and lane', () => {
	const tasks: VizTask[] = [
		{
			task_id: 'a',
			parent_task_id: 'mr1',
			agent_type: 'mr2',
			status: 'running',
			description: 'a',
			started_at: '2024-01-01T00:00:00+00:00',
			updated_at: null,
			finished_at: null,
			pid: 1,
			event_count: 1,
			path: [0],
			lane: 'conversation'
		},
		{
			task_id: 'b',
			parent_task_id: 'mr1',
			agent_type: 'mem_dltr',
			status: 'completed',
			description: 'b',
			started_at: '2024-01-01T00:00:00+00:00',
			updated_at: null,
			finished_at: '2024-01-01T00:00:02+00:00',
			pid: 2,
			event_count: 1,
			path: [0],
			lane: 'system'
		}
	];
	const rows = buildTaskRowMap(tasks, 10, 24);
	assert.ok((rows.get('a') ?? 0) > 10);
	assert.ok((rows.get('b') ?? 99) < 10);
});
