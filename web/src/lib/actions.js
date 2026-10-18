// Movement vocabulary shared with the backend (movement_schema.py,
// repetition_counter_v3.py).

export const ACTION_LABELS = {
  squat: 'Squat',
  lunge: 'Lunge',
  deadlift: 'Deadlift',
  push_up: 'Push-up',
  bench_press: 'Bench press',
  shoulder_press: 'Shoulder press',
  bicep_curl: 'Bicep curl',
  pull_up: 'Pull-up',
  row: 'Row',
  jumping_jack: 'Jumping jack',
  unknown: 'Unknown',
};

// The ten movements the engine can count, in the order the picker shows them.
export const REP_ACTIONS = [
  'squat', 'lunge', 'deadlift', 'push_up', 'bench_press',
  'shoulder_press', 'bicep_curl', 'pull_up', 'row', 'jumping_jack',
];

export const REP_GOALS = {
  squat: 15, lunge: 12, deadlift: 10, push_up: 12, bench_press: 10,
  shoulder_press: 10, bicep_curl: 12, pull_up: 8, row: 10, jumping_jack: 20,
};

// Rough energy cost per rep in kcal, for the session estimate only.
const KCAL_PER_REP = {
  squat: 0.65, lunge: 0.55, deadlift: 0.75, push_up: 0.45, bench_press: 0.42,
  shoulder_press: 0.42, bicep_curl: 0.32, pull_up: 0.6, row: 0.45, jumping_jack: 0.18,
};

export const label = (action) => ACTION_LABELS[action] || String(action || '').replace(/_/g, ' ');

// The server reports 'home' (start position) and 'effort' (working position).
export function stageLabel(stage) {
  if (stage === 'home') return 'Start';
  if (stage === 'effort') return 'Effort';
  return 'Rest';
}

export function estimateKcal(counters, seconds) {
  const reps = Object.entries(counters || {}).reduce(
    (sum, [action, count]) => sum + (KCAL_PER_REP[action] || 0.4) * Number(count || 0),
    0,
  );
  return Math.round(reps + seconds * 0.05);
}

export const STATUS_TEXT = {
  'Buffering pose window': 'Reading your movement',
  'Stabilizing movement': 'Confirming the exercise',
  'Low-confidence movement': 'Movement unclear, holding the last exercise',
  'Pose gap; rebuilding window': 'Tracking dropped, rebuilding',
  'No visible pose': 'No one in frame. Step back so your whole body shows.',
  'Partial pose: keep one full side of the body in view': 'Keep one full side of your body in view',
  'Invalid image payload': 'That frame could not be read',
};

// Short form cues; 'effort' means the working end of the movement.
const CUES = {
  squat: { home: 'Chest up, sit your hips back and down.', effort: 'Good depth. Drive through your heels to stand.' },
  lunge: { home: 'Long stride, torso tall, step down with control.', effort: 'Back knee near the floor. Push up through the front heel.' },
  deadlift: { home: 'Brace, hinge at the hips, keep the bar close.', effort: 'Flat back at the bottom. Stand up by pushing the floor away.' },
  push_up: { home: 'Body in one straight line, hands under shoulders.', effort: 'Chest near the floor. Press back up without sagging.' },
  bench_press: { home: 'Shoulder blades pinned, lower the bar to mid-chest.', effort: 'Touch and press. Keep wrists stacked over elbows.' },
  shoulder_press: { home: 'Ribs down. Lower the weight to shoulder height.', effort: 'Press straight up and finish with arms locked.' },
  bicep_curl: { home: 'Elbows pinned to your sides. Curl without swinging.', effort: 'Squeeze at the top, then lower slowly.' },
  pull_up: { home: 'Hang long, then pull your elbows down to your ribs.', effort: 'Chin over the bar. Lower all the way with control.' },
  row: { home: 'Hinge, flat back, pull toward your lower ribs.', effort: 'Squeeze the shoulder blades, then lower with control.' },
  jumping_jack: { home: 'Stay light on your feet and keep the rhythm.', effort: 'Arms overhead, land softly.' },
};

export function cueFor(action, stage) {
  const cues = CUES[action];
  if (!cues) return null;
  return stage === 'effort' ? cues.effort : cues.home;
}
