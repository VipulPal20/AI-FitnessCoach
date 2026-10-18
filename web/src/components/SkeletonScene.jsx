import { Canvas, useFrame } from '@react-three/fiber';
import { useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import { POSE_CONNECTIONS, jointAngle } from '../lib/pose.js';

// Joints drawn in the figure (face points other than the nose are skipped).
const BODY_JOINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32];
const FOOT_JOINTS = [27, 28, 29, 30, 31, 32];
const BONES = [...POSE_CONNECTIONS.filter(([a, b]) => a >= 11 && b >= 11 && a <= 32 && b <= 32 && ![17, 18, 19, 20, 21, 22].includes(b)), ['neck', 0]];
const UP = new THREE.Vector3(0, 1, 0);

// Same rule shape as repetition_counter_v3.py: home -> effort -> home is a rep.
// Thresholds differ from the server's 2D ones: these are 3D world-landmark
// angles, where smoothing softens full lockout (standing reads ~157°) and the
// hip hinge isn't foreshortened by a front-on camera.
export const MOVES = {
  squat: { joint: [23, 25, 27], home: 150, effort: 100, label: 'Knee' },
  deadlift: { joint: [11, 23, 25], home: 150, effort: 100, label: 'Hip' },
};

/**
 * Converts MediaPipe world landmarks (metres, hip-centred, y down) into scene
 * space (y up) and plants the feet on a floor, so the hips travel instead of
 * the feet floating.
 */
function prepare(frames) {
  const flipped = frames.map((joints) => joints.map(([x, y, z]) => [x, -y, -z]));
  const floor = Math.min(...flipped.map((joints) => Math.min(...FOOT_JOINTS.map((j) => joints[j][1]))));
  return flipped.map((joints) => {
    const lowest = Math.min(...FOOT_JOINTS.map((j) => joints[j][1]));
    const lift = floor - lowest;
    return joints.map(([x, y, z]) => [x, y + lift - floor, z]);
  });
}

function Figure({ clip, fps, onSample, pointer }) {
  const group = useRef();
  const jointRefs = useRef({});
  const boneRefs = useRef([]);
  const clock = useRef(0);
  const tmp = useMemo(() => ({ a: new THREE.Vector3(), b: new THREE.Vector3(), mid: new THREE.Vector3(), dir: new THREE.Vector3() }), []);

  useFrame((_, delta) => {
    if (!clip?.length) return;
    clock.current += Math.min(delta, 0.1);
    const position = (clock.current * fps) % clip.length;
    const i = Math.floor(position);
    const t = position - i;
    const A = clip[i];
    const B = clip[(i + 1) % clip.length];
    const at = (j) => [A[j][0] + (B[j][0] - A[j][0]) * t, A[j][1] + (B[j][1] - A[j][1]) * t, A[j][2] + (B[j][2] - A[j][2]) * t];

    const points = {};
    BODY_JOINTS.forEach((j) => {
      points[j] = at(j);
      jointRefs.current[j]?.position.set(...points[j]);
    });
    points.neck = points[11].map((v, k) => (v + points[12][k]) / 2);

    BONES.forEach(([p, q], index) => {
      const mesh = boneRefs.current[index];
      if (!mesh) return;
      tmp.a.set(...points[p]);
      tmp.b.set(...points[q]);
      tmp.mid.addVectors(tmp.a, tmp.b).multiplyScalar(0.5);
      tmp.dir.subVectors(tmp.b, tmp.a);
      const length = tmp.dir.length();
      mesh.position.copy(tmp.mid);
      mesh.quaternion.setFromUnitVectors(UP, tmp.dir.normalize());
      mesh.scale.set(1, length, 1);
    });

    if (group.current) {
      // Turn toward the cursor, with a slow idle sway.
      const targetY = pointer.current.x * 0.9 + Math.sin(clock.current * 0.35) * 0.25;
      const targetX = pointer.current.y * 0.15;
      group.current.rotation.y += (targetY - group.current.rotation.y) * 0.06;
      group.current.rotation.x += (targetX - group.current.rotation.x) * 0.06;
    }
    onSample?.(points);
  });

  return (
    <group ref={group}>
      {BODY_JOINTS.map((j) => (
        <mesh key={j} ref={(m) => { jointRefs.current[j] = m; }}>
          <sphereGeometry args={[j === 0 ? 0.075 : 0.032, 24, 24]} />
          <meshStandardMaterial color={j === 0 ? '#f3f3ef' : '#c6f432'} roughness={0.35} />
        </mesh>
      ))}
      {BONES.map((_, index) => (
        <mesh key={index} ref={(m) => { boneRefs.current[index] = m; }}>
          <cylinderGeometry args={[0.016, 0.016, 1, 12]} />
          <meshStandardMaterial color="#f3f3ef" roughness={0.5} />
        </mesh>
      ))}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.001, 0]}>
        <circleGeometry args={[0.62, 64]} />
        <meshBasicMaterial color="#2a2d2a" />
      </mesh>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.002, 0]}>
        <ringGeometry args={[0.6, 0.62, 64]} />
        <meshBasicMaterial color="#c6f432" />
      </mesh>
    </group>
  );
}

// Camera framing: R3F aims a new camera at the origin (the floor here) unless a
// rotation is given, so tilt it to look at the middle of a standing body,
// CAMERA_AIM metres up, from the very first frame.
const CAMERA_POSITION = [0, 0.95, 3.6];
const CAMERA_AIM = 0.72;
const CAMERA_TILT = -Math.atan2(CAMERA_POSITION[1] - CAMERA_AIM, CAMERA_POSITION[2]);

/**
 * Replays a captured movement clip as a 3D skeleton and reports the tracked
 * joint angle, stage and rep count back to the caller every frame.
 */
export default function SkeletonScene({ move = 'squat', onReadout, paused = false }) {
  const pointer = useRef({ x: 0, y: 0 });
  const cache = useRef({});
  const [clip, setClip] = useState(null);
  const repState = useRef({ stage: null, reps: 0, lastEmit: 0 });

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      if (!cache.current[move]) {
        const response = await fetch(`/motion/${move}.json`);
        const data = await response.json();
        cache.current[move] = { frames: prepare(data.frames), fps: data.fps, move };
      }
      if (cancelled) return;
      repState.current = { stage: null, reps: 0, lastEmit: 0 };
      setClip(cache.current[move]);
    };
    load().catch(() => {});
    return () => { cancelled = true; };
  }, [move]);

  const onSample = (points) => {
    if (!clip) return;
    const rule = MOVES[clip.move];
    const [a, b, c] = rule.joint;
    const angle = jointAngle(points[a], points[b], points[c]);
    const state = repState.current;
    if (angle !== null) {
      if (state.stage === null && angle >= rule.home) state.stage = 'home';
      else if (state.stage === 'home' && angle <= rule.effort) state.stage = 'effort';
      else if (state.stage === 'effort' && angle >= rule.home) { state.stage = 'home'; state.reps += 1; }
    }
    const now = performance.now();
    if (onReadout && now - state.lastEmit > 90) {
      state.lastEmit = now;
      onReadout({ angle, stage: state.stage, reps: state.reps, label: rule.label });
    }
  };

  const onPointerMove = (event) => {
    const box = event.currentTarget.getBoundingClientRect();
    pointer.current.x = ((event.clientX - box.left) / box.width - 0.5) * 2;
    pointer.current.y = ((event.clientY - box.top) / box.height - 0.5) * 2;
  };

  return (
    <div className="skeleton-scene" onPointerMove={onPointerMove} onPointerLeave={() => { pointer.current = { x: 0, y: 0 }; }}>
      <Canvas
        camera={{ position: CAMERA_POSITION, rotation: [CAMERA_TILT, 0, 0], fov: 38 }}
        dpr={[1, 2]}
        frameloop={paused ? 'never' : 'always'}
        gl={{ antialias: true, alpha: true }}
        aria-hidden="true"
      >
        <ambientLight intensity={0.75} />
        <directionalLight position={[2, 3, 2]} intensity={1.4} />
        <directionalLight position={[-2, 1, -1]} intensity={0.4} />
        {clip && <Figure key={clip.move} clip={clip.frames} fps={clip.fps} onSample={onSample} pointer={pointer} />}
      </Canvas>
    </div>
  );
}
