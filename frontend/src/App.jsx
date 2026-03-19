import * as React from 'react'
import { Canvas, useThree } from '@react-three/fiber'
import { Line, OrbitControls, Text } from '@react-three/drei'
import * as THREE from 'three'

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
const WS_BASE = API_BASE.replace(/^http/, 'ws')

const TRANSLATION_PARAMS = [
  { key: 'tx', label: 'Tx', step: 1, unit: 'mm', group: 'translation' },
  { key: 'ty', label: 'Ty', step: 1, unit: 'mm', group: 'translation' },
  { key: 'tz', label: 'Tz', step: 1, unit: 'mm', group: 'translation' },
]

const ROTATION_PARAMS = [
  { key: 'rx', label: 'Rx', min: -180, max: 180, step: 1, unit: 'deg', group: 'rotation' },
  { key: 'ry', label: 'Ry', min: -180, max: 180, step: 1, unit: 'deg', group: 'rotation' },
  { key: 'rz', label: 'Rz', min: -180, max: 180, step: 1, unit: 'deg', group: 'rotation' },
]

const PARAM_CONFIG = [...TRANSLATION_PARAMS, ...ROTATION_PARAMS]

const INITIAL_POSE = { tx: 0, ty: 0, tz: 0, rx: 0, ry: 0, rz: 0 }

const AXIS_COLORS = { x: '#ff5a4f', y: '#4caf50', z: '#2f7de1' }

const PLACEHOLDER_VIEWS = [{ view: 'AP' }]

const FRAME_CAMERA_DEFAULT = { position: [10, 6, 12], target: [0, 0, -2], fov: 48 }

// ---------------------------------------------------------------------------
// Math helpers — mirrors backend's _euler_to_rotation / _apply_pose exactly
// ---------------------------------------------------------------------------
const toRad = (deg) => (deg * Math.PI) / 180

function eulerToRotation(rx, ry, rz) {
  const ax = toRad(rx), ay = toRad(ry), az = toRad(rz)
  const sx = Math.sin(ax), cx = Math.cos(ax)
  const sy = Math.sin(ay), cy = Math.cos(ay)
  const sz = Math.sin(az), cz = Math.cos(az)
  // R = Rz * Ry * Rx — same as backend
  return [
    [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
    [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
    [-sy, cy * sx, cy * cx],
  ]
}

const mat3MulVec = (m, v) => [
  m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
  m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
  m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
]

const transpose3 = (m) => [
  [m[0][0], m[1][0], m[2][0]],
  [m[0][1], m[1][1], m[2][1]],
  [m[0][2], m[1][2], m[2][2]],
]

const mat3Mul = (a, b) => [
  [a[0][0]*b[0][0]+a[0][1]*b[1][0]+a[0][2]*b[2][0], a[0][0]*b[0][1]+a[0][1]*b[1][1]+a[0][2]*b[2][1], a[0][0]*b[0][2]+a[0][1]*b[1][2]+a[0][2]*b[2][2]],
  [a[1][0]*b[0][0]+a[1][1]*b[1][0]+a[1][2]*b[2][0], a[1][0]*b[0][1]+a[1][1]*b[1][1]+a[1][2]*b[2][1], a[1][0]*b[0][2]+a[1][1]*b[1][2]+a[1][2]*b[2][2]],
  [a[2][0]*b[0][0]+a[2][1]*b[1][0]+a[2][2]*b[2][0], a[2][0]*b[0][1]+a[2][1]*b[1][1]+a[2][2]*b[2][1], a[2][0]*b[0][2]+a[2][1]*b[1][2]+a[2][2]*b[2][2]],
]

const vecSub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
const vecAdd = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]]

/**
 * Replicate backend's camera-relative _apply_pose.
 * Perturbations are in the camera's own coordinate frame.
 * sceneInfo.camera.basis.{x,y,z} are columns of R_c2w_default (= default_R^T).
 */
function applyPose(sceneInfo, pose) {
  const { camera, volume } = sceneInfo
  const centroid = volume.centroid
  const defaultSource = camera.source

  // Reconstruct default_R (world-to-camera) from basis (camera-to-world columns)
  // R_c2w_default columns = camera.basis.{x, y, z}
  // default_R = R_c2w_default^T → rows of default_R = basis vectors
  const R_c2w_default = [
    [camera.basis.x[0], camera.basis.y[0], camera.basis.z[0]],
    [camera.basis.x[1], camera.basis.y[1], camera.basis.z[1]],
    [camera.basis.x[2], camera.basis.y[2], camera.basis.z[2]],
  ]
  const default_R = transpose3(R_c2w_default)

  // Camera-relative rotation → conjugate to world frame
  // R_world = R_c2w_default @ R_local @ default_R
  const R_local = eulerToRotation(pose.rx, pose.ry, pose.rz)
  const R_world = mat3Mul(R_c2w_default, mat3Mul(R_local, default_R))

  // Orbit source around centroid
  const sourceRel = vecSub(defaultSource, centroid)
  const sourceRotated = vecAdd(mat3MulVec(R_world, sourceRel), centroid)

  // Camera-relative translation → convert to world via posed R_c2w
  // R_w2c = default_R @ R_world^T; R_c2w = R_w2c^T
  const R_w2c = mat3Mul(default_R, transpose3(R_world))
  const R_c2w = transpose3(R_w2c)
  const t_cam = [pose.tx, pose.ty, pose.tz]
  const t_world = mat3MulVec(R_c2w, t_cam)

  const source = vecAdd(sourceRotated, t_world)

  // Cam-to-world basis columns from R_c2w
  const basisX = [R_c2w[0][0], R_c2w[1][0], R_c2w[2][0]]
  const basisY = [R_c2w[0][1], R_c2w[1][1], R_c2w[2][1]]
  const basisZ = [R_c2w[0][2], R_c2w[1][2], R_c2w[2][2]]

  return { source, basis: { x: basisX, y: basisY, z: basisZ } }
}

// ---------------------------------------------------------------------------
// World coords (mm) → Three.js scene coords
// Backend world: X=right, Y=anterior (into patient), Z=superior (up)
// Three.js: X=right, Y=up, Z=toward viewer
// Mapping: scene_x = world_x, scene_y = world_z, scene_z = -world_y
// ---------------------------------------------------------------------------
const WORLD_SCALE = 0.005 // mm to scene units (volume ~500mm → ~2.5 scene units)

const worldToScene = (v) => [v[0] * WORLD_SCALE, v[2] * WORLD_SCALE, -v[1] * WORLD_SCALE]
const worldToSceneVec = (v) => new THREE.Vector3(v[0], v[2], -v[1])

function basisToQuaternion(basis) {
  const bx = worldToSceneVec(basis.x).normalize()
  const by = worldToSceneVec(basis.y).normalize()
  const bz = worldToSceneVec(basis.z).normalize()

  const matrix = new THREE.Matrix4().set(
    bx.x, by.x, bz.x, 0,
    bx.y, by.y, bz.y, 0,
    bx.z, by.z, bz.z, 0,
    0, 0, 0, 1,
  )
  return new THREE.Quaternion().setFromRotationMatrix(matrix)
}

// ---------------------------------------------------------------------------
// 3D Components
// ---------------------------------------------------------------------------
function AxisTriad({ origin, basis, length = 3.1, lengths = null, dashed = false, opacity = 1, labels = null }) {
  const axes = React.useMemo(() => {
    const start = new THREE.Vector3(...worldToScene(origin))
    const up = new THREE.Vector3(0, 1, 0)

    return [
      { key: 'x', color: AXIS_COLORS.x, dir: worldToSceneVec(basis.x).normalize() },
      { key: 'y', color: AXIS_COLORS.y, dir: worldToSceneVec(basis.y).normalize() },
      { key: 'z', color: AXIS_COLORS.z, dir: worldToSceneVec(basis.z).normalize() },
    ].map((entry) => {
      const len = lengths?.[entry.key] ?? length
      const end = start.clone().addScaledVector(entry.dir, len)
      const tipQ = new THREE.Quaternion().setFromUnitVectors(up, entry.dir)
      const labelPos = start.clone().addScaledVector(entry.dir, len + 0.7)
      return { ...entry, start, end, tipPos: end.toArray(), tipQ, labelPos: labelPos.toArray() }
    })
  }, [origin, basis, length, lengths])

  return (
    <group>
      {axes.map((axis) => (
        <group key={axis.key}>
          <Line
            points={[axis.start.toArray(), axis.end.toArray()]}
            color={axis.color}
            lineWidth={1.7}
            dashed={dashed}
            dashSize={0.24}
            gapSize={0.16}
            transparent
            opacity={opacity}
          />
          <mesh position={axis.tipPos} quaternion={axis.tipQ}>
            <coneGeometry args={[0.15, 0.42, 14]} />
            <meshStandardMaterial color={axis.color} transparent opacity={opacity} />
          </mesh>
          {labels?.[axis.key] && (
            <Text
              position={axis.labelPos}
              fontSize={1.1}
              color={axis.color}
              anchorX="center"
              anchorY="middle"
              outlineColor="#ffffff"
              outlineWidth={0.12}
              fontWeight="bold"
            >
              {labels[axis.key]}
            </Text>
          )}
        </group>
      ))}
      <mesh position={worldToScene(origin)}>
        <sphereGeometry args={[0.13, 20, 20]} />
        <meshStandardMaterial color="#d88b00" />
      </mesh>
    </group>
  )
}


function CTVolume({ origin, halfExtent }) {
  const pos = React.useMemo(() => worldToScene(origin), [origin])
  // halfExtent is [hx, hy, hz] in scene units
  const hx = halfExtent[0], hy = halfExtent[1], hz = halfExtent[2]

  const { corners, edges } = React.useMemo(() => {
    const c = [
      [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
      [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],
    ]
    return {
      corners: c,
      edges: [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
      ],
    }
  }, [hx, hy, hz])

  return (
    <group position={pos}>
      <mesh>
        <boxGeometry args={[hx * 2, hy * 2, hz * 2]} />
        <meshStandardMaterial color="#e56700" transparent opacity={0.08} depthWrite={false} />
      </mesh>
      {edges.map(([a, b], idx) => (
        <Line
          key={`ct-edge-${idx}`}
          points={[corners[a], corners[b]]}
          color="#e56700"
          lineWidth={1.25}
          dashed
          dashSize={0.2}
          gapSize={0.12}
          transparent
          opacity={0.82}
        />
      ))}
      <Text
        position={[hx + 0.62, -hy * 0.48, hz * 0.74]}
        fontSize={0.32}
        color="#a24800"
        anchorX="center"
        anchorY="middle"
        outlineColor="#fff7ef"
        outlineWidth={0.02}
      >
        CT
      </Text>
    </group>
  )
}

function APCameraModel({ origin, basis }) {
  const position = React.useMemo(() => worldToScene(origin), [origin])
  const quaternion = React.useMemo(() => basisToQuaternion(basis), [basis])

  const frustum = React.useMemo(() => {
    const near = -1.8
    const hw = 0.82
    const hh = 0.56
    return [[-hw, -hh, near], [hw, -hh, near], [hw, hh, near], [-hw, hh, near]]
  }, [])

  return (
    <group position={position} quaternion={quaternion} scale={[0.74, 0.74, 0.74]}>
      <mesh position={[0, 0, 0]}>
        <boxGeometry args={[0.92, 0.58, 0.62]} />
        <meshStandardMaterial color="#2f80df" metalness={0.1} roughness={0.35} />
      </mesh>
      <mesh position={[0, 0, -0.45]} rotation={[Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[0.18, 0.18, 0.24, 20]} />
        <meshStandardMaterial color="#f5f9ff" metalness={0.04} roughness={0.28} />
      </mesh>
      <mesh position={[0, 0.28, 0.04]}>
        <boxGeometry args={[0.32, 0.09, 0.22]} />
        <meshStandardMaterial color="#1b58a0" />
      </mesh>
      <mesh position={[0, 0, -1.8]}>
        <planeGeometry args={[1.64, 1.12]} />
        <meshStandardMaterial color="#4f9cf1" transparent opacity={0.2} side={THREE.DoubleSide} />
      </mesh>
      <Line points={[frustum[0], frustum[1], frustum[2], frustum[3], frustum[0]]} color="#4d93e2" lineWidth={1.2} />
      {frustum.map((corner, idx) => (
        <Line key={`frustum-ray-${idx}`} points={[[0, 0, -0.12], corner]} color="#4d93e2" lineWidth={1.1} />
      ))}
    </group>
  )
}

// ---------------------------------------------------------------------------
// Auto-fit: adjust orbit controls to encompass camera + CT volume
// resetKey change → pick a fresh isometric angle; pose-only change → keep user orbit direction
// ---------------------------------------------------------------------------
function AutoFitControls({ controlsRef, cameraScenePos, volumeScenePos, resetKey }) {
  const { camera } = useThree()
  const lastResetKey = React.useRef(-1)

  React.useEffect(() => {
    const controls = controlsRef.current
    if (!controls) return

    const camPos = new THREE.Vector3(...cameraScenePos)
    const volPos = new THREE.Vector3(...volumeScenePos)

    const center = new THREE.Vector3().addVectors(camPos, volPos).multiplyScalar(0.5)
    const radius = camPos.distanceTo(volPos) / 2 + 1.5

    const fov = camera.fov * (Math.PI / 180)
    const dist = radius / Math.sin(fov / 2) * 1.15

    // Fresh angle on first mount (lastResetKey starts at -1) or explicit reset
    const forceAngle = resetKey !== lastResetKey.current
    lastResetKey.current = resetKey

    if (forceAngle) {
      const dir = new THREE.Vector3(0.6, 0.4, 0.7).normalize()
      camera.position.copy(center.clone().addScaledVector(dir, dist))
    } else {
      const dir = new THREE.Vector3().subVectors(camera.position, controls.target).normalize()
      camera.position.copy(center.clone().addScaledVector(dir, dist))
    }

    controls.target.copy(center)
    controls.update()
  }, [cameraScenePos, volumeScenePos, resetKey, camera, controlsRef])

  return null
}

// ---------------------------------------------------------------------------
// FrameScene — uses backend scene info, no hardcoded camera geometry
// ---------------------------------------------------------------------------
function FrameScene({ sceneInfo, pose }) {
  const identityBasis = { x: [1, 0, 0], y: [0, 1, 0], z: [0, 0, 1] }

  if (!sceneInfo) {
    return (
      <>
        <ambientLight intensity={0.78} />
        <directionalLight position={[16, 22, 12]} intensity={1.1} />
      </>
    )
  }

  // Compute posed camera from backend scene info + current pose
  const posedCamera = applyPose(sceneInfo, pose)

  // Volume centroid and half-extent in scene coords
  const volCentroid = sceneInfo.volume.centroid
  const volExtent = sceneInfo.volume.extent
  // Half-extent mapped through worldToScene: world [ex, ey, ez] → scene [ex, ez, ey]
  const halfExtentScene = [
    volExtent[0] * WORLD_SCALE / 2,
    volExtent[2] * WORLD_SCALE / 2,
    volExtent[1] * WORLD_SCALE / 2,
  ]

  return (
    <>
      <ambientLight intensity={0.78} />
      <directionalLight position={[16, 22, 12]} intensity={1.1} />
      <directionalLight position={[-12, 9, -16]} intensity={0.45} />

      <CTVolume origin={volCentroid} halfExtent={halfExtentScene} />
      <APCameraModel origin={posedCamera.source} basis={posedCamera.basis} />

      <AxisTriad
        origin={volCentroid}
        basis={identityBasis}
        lengths={{ x: halfExtentScene[0], y: halfExtentScene[2], z: halfExtentScene[1] }}
        opacity={0.95}
        labels={{ x: 'L', y: 'P', z: 'S' }}
      />
      <AxisTriad origin={posedCamera.source} basis={posedCamera.basis} length={2.4} opacity={0.9} />
    </>
  )
}

function FrameIllustration({ pose, sceneInfo }) {
  const controlsRef = React.useRef(null)
  const [resetKey, setResetKey] = React.useState(0)

  // Compute scene positions for auto-fit
  const positions = React.useMemo(() => {
    if (!sceneInfo) return null
    const posedCamera = applyPose(sceneInfo, pose)
    return {
      camera: worldToScene(posedCamera.source),
      volume: worldToScene(sceneInfo.volume.centroid),
    }
  }, [sceneInfo, pose])

  const resetView = () => setResetKey(k => k + 1)

  return (
    <section className="frame-panel">
      <div className="panel-title-row">
        <div className="panel-heading">
          <h2>Scene View</h2>
        </div>
        <div className="panel-tools">
          <button type="button" className="ghost-btn mini" onClick={resetView}>
            Reset View
          </button>
          <span className="panel-chip">3D Live</span>
        </div>
      </div>

      <div className="frame-canvas-wrap" role="img" aria-label="Interactive 3D frame widget">
        <Canvas camera={{ position: FRAME_CAMERA_DEFAULT.position, fov: FRAME_CAMERA_DEFAULT.fov }} dpr={[1, 2]}>
          <color attach="background" args={['#ffffff']} />
          <FrameScene sceneInfo={sceneInfo} pose={pose} />
          <OrbitControls
            ref={controlsRef}
            makeDefault
            enableDamping
            dampingFactor={0.12}
            rotateSpeed={0.9}
            panSpeed={0.7}
            minDistance={1}
            maxDistance={50}
            target={FRAME_CAMERA_DEFAULT.target}
          />
          {positions && (
            <AutoFitControls
              controlsRef={controlsRef}
              cameraScenePos={positions.camera}
              volumeScenePos={positions.volume}
              resetKey={resetKey}
            />
          )}
        </Canvas>
      </div>
    </section>
  )
}

// ---------------------------------------------------------------------------
// HoldButton — fires immediately, then accelerates while held
// ---------------------------------------------------------------------------
function HoldButton({ onStep, children, className, 'aria-label': ariaLabel, rotationScale = false }) {
  const timerRef = React.useRef(null)
  const startTimeRef = React.useRef(0)

  const stop = React.useCallback(() => {
    if (timerRef.current) {
      clearTimeout(timerRef.current)
      timerRef.current = null
    }
  }, [])

  const schedule = React.useCallback(() => {
    const elapsed = Date.now() - startTimeRef.current
    const steps = rotationScale
      ? (elapsed < 600 ? 0.1 : elapsed < 1500 ? 0.5 : elapsed < 3000 ? 2 : 10)
      : (elapsed < 600 ? 1    : elapsed < 1500 ? 5   : elapsed < 3000 ? 20  : 100)
    const interval = Math.max(30, 80 - elapsed * 0.018)
    timerRef.current = setTimeout(() => {
      onStep(steps)
      schedule()
    }, interval)
  }, [onStep, rotationScale])

  const initialStep = rotationScale ? 0.1 : 1

  const start = React.useCallback(() => {
    stop()
    startTimeRef.current = Date.now()
    onStep(initialStep)
    timerRef.current = setTimeout(schedule, 250)
  }, [onStep, initialStep, schedule, stop])

  React.useEffect(() => stop, [stop]) // cleanup on unmount

  return (
    <button
      type="button"
      className={className}
      aria-label={ariaLabel}
      onPointerDown={(e) => { e.preventDefault(); start() }}
      onPointerUp={stop}
      onPointerLeave={stop}
      onPointerCancel={stop}
    >
      {children}
    </button>
  )
}

// ---------------------------------------------------------------------------
// Pose Controls
// ---------------------------------------------------------------------------
function TranslationAxisCard({ param, value, onChange }) {
  const STEP = 1

  const handleNumber = (event) => {
    const next = Number(event.target.value)
    if (Number.isNaN(next)) return
    onChange(param.key, next)
  }

  return (
    <div className="pose-row translation">
      <div className="pose-row-label">
        <span className="pose-axis-dot translation" />
        <span className="pose-axis-label">{param.label}</span>
      </div>
      <HoldButton
        className="axis-step-btn"
        aria-label={`Decrease ${param.label}`}
        onStep={(s) => onChange(param.key, (prev) => prev - s)}
      >−</HoldButton>
      <input
        className="pose-axis-value translation-value"
        type="number"
        step={param.step}
        value={Math.round(value * 10) / 10}
        onChange={handleNumber}
        aria-label={`${param.label} value`}
      />
      <HoldButton
        className="axis-step-btn"
        aria-label={`Increase ${param.label}`}
        onStep={(s) => onChange(param.key, (prev) => prev + s)}
      >+</HoldButton>
      <button type="button" className="axis-zero-btn" onClick={() => onChange(param.key, 0)} aria-label={`Zero ${param.label}`}>0</button>
    </div>
  )
}

function RotationAxisCard({ param, value, onChange }) {
  const step = 0.1
  const clampRot = (v) => Math.max(param.min, Math.min(param.max, Math.round(v * 100) / 100))

  const handleNumber = (event) => {
    const next = Number(event.target.value)
    if (Number.isNaN(next)) return
    onChange(param.key, clampRot(next))
  }

  return (
    <div className="pose-row rotation">
      <div className="pose-row-label">
        <span className="pose-axis-dot rotation" />
        <span className="pose-axis-label">{param.label}</span>
      </div>
      <HoldButton
        className="axis-step-btn"
        aria-label={`Decrease ${param.label}`}
        onStep={(s) => onChange(param.key, (prev) => clampRot(prev - s))}
        rotationScale
      >−</HoldButton>
      <input
        className="pose-axis-value"
        type="number"
        min={param.min}
        max={param.max}
        step={step}
        value={Math.round(value * 100) / 100}
        onChange={handleNumber}
        aria-label={`${param.label} value`}
      />
      <HoldButton
        className="axis-step-btn"
        aria-label={`Increase ${param.label}`}
        onStep={(s) => onChange(param.key, (prev) => clampRot(prev + s))}
        rotationScale
      >+</HoldButton>
      <button type="button" className="axis-zero-btn" onClick={() => onChange(param.key, 0)} aria-label={`Zero ${param.label}`}>0</button>
    </div>
  )
}

function PoseControlsPanel({
  pose, onChangePose, onResetGroup, onResetPose,
  preset, availablePresets, onChangePreset,
  onFetchTransform, extrinsicMatrix, onDismissTransform,
  onOpenIntrinsics, intrinsicsOpen, intrinsicsData, onIntrinsicsChange,
  onApplyIntrinsics, onResetIntrinsics, onDismissIntrinsics,
}) {
  return (
    <section className="pose-panel">
      <header className="pose-panel-head">
        <h2>Camera Pose</h2>
        <button type="button" className="ghost-btn tiny" onClick={onResetPose}>Reset</button>
      </header>

      <div className="preset-row">
        <label className="preset-label">Preset</label>
        <select
          className="preset-select"
          value={preset}
          onChange={(e) => onChangePreset(e.target.value)}
        >
          {availablePresets.map((p) => (
            <option key={p} value={p}>{p}</option>
          ))}
        </select>
      </div>

      <section className="axis-group">
        <header className="axis-group-head">
          <h3>Translation (mm)</h3>
          <button type="button" className="ghost-btn tiny" onClick={() => onResetGroup('translation')}>Zero T</button>
        </header>
        {TRANSLATION_PARAMS.map((param) => (
          <TranslationAxisCard key={param.key} param={param} value={pose[param.key]} onChange={onChangePose} />
        ))}
      </section>

      <section className="axis-group">
        <header className="axis-group-head">
          <h3>Rotation (deg)</h3>
          <button type="button" className="ghost-btn tiny" onClick={() => onResetGroup('rotation')}>Zero R</button>
        </header>
        {ROTATION_PARAMS.map((param) => (
          <RotationAxisCard key={param.key} param={param} value={pose[param.key]} onChange={onChangePose} />
        ))}
      </section>

      <p className="pose-convention">All offsets are in the camera's local frame.</p>

      <button type="button" className="ghost-btn transform-btn" onClick={onFetchTransform}>
        Get World-to-Camera Transform
      </button>

      {extrinsicMatrix && (
        <div className="modal-overlay" onClick={(e) => { if (e.target === e.currentTarget) onDismissTransform() }}>
          <div className="modal-card">
            <div className="modal-head">
              <h3>4×4 Extrinsic Matrix (world → camera)</h3>
              <button type="button" className="ghost-btn tiny" onClick={onDismissTransform}>✕</button>
            </div>
            <pre className="extrinsic-matrix">
              {extrinsicMatrix.map((row, i) => {
                const inner = row.map((v) => v.toFixed(6).padStart(12)).join(', ')
                const prefix = i === 0 ? '[[' : ' ['
                const suffix = i < extrinsicMatrix.length - 1 ? '],' : ']]'
                return prefix + inner + suffix
              }).join('\n')}
            </pre>
            <div className="modal-actions">
              <button
                type="button"
                className="ghost-btn mini"
                onClick={() => {
                  const text = extrinsicMatrix.map((row, i) => {
                    const inner = row.map((v) => v.toFixed(6).padStart(12)).join(', ')
                    const prefix = i === 0 ? '[[' : ' ['
                    const suffix = i < extrinsicMatrix.length - 1 ? '],' : ']]'
                    return prefix + inner + suffix
                  }).join('\n')
                  navigator.clipboard.writeText(text)
                }}
              >Copy to Clipboard</button>
              <button type="button" className="ghost-btn mini" onClick={onDismissTransform}>Close</button>
            </div>
          </div>
        </div>
      )}

      <button type="button" className="ghost-btn transform-btn" onClick={onOpenIntrinsics}>
        Edit Intrinsics
      </button>

      {intrinsicsOpen && intrinsicsData && (
        <div className="modal-overlay" onClick={(e) => { if (e.target === e.currentTarget) onDismissIntrinsics() }}>
          <div className="modal-card">
            <div className="modal-head">
              <h3>Camera Intrinsics (K)</h3>
              <button type="button" className="ghost-btn tiny" onClick={onDismissIntrinsics}>✕</button>
            </div>
            <div className="intrinsics-grid">
              {['fx', 'fy', 'cx', 'cy'].map((field) => (
                <label key={field} className="intrinsics-field">
                  <span className="intrinsics-field-label">{field}</span>
                  <input
                    className="intrinsics-field-input"
                    type="number"
                    step="0.1"
                    value={intrinsicsData[field]}
                    onChange={(e) => {
                      const v = Number(e.target.value)
                      if (!Number.isNaN(v)) onIntrinsicsChange({ ...intrinsicsData, [field]: v })
                    }}
                  />
                </label>
              ))}
            </div>
            <pre className="extrinsic-matrix">
              {(() => {
                const { fx, fy, cx, cy } = intrinsicsData
                const K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
                return K.map((row, i) => {
                  const inner = row.map((v) => v.toFixed(4).padStart(12)).join(', ')
                  const prefix = i === 0 ? '[[' : ' ['
                  const suffix = i < K.length - 1 ? '],' : ']]'
                  return prefix + inner + suffix
                }).join('\n')
              })()}
            </pre>
            <div className="modal-actions">
              <button type="button" className="ghost-btn mini" onClick={onApplyIntrinsics}>Apply</button>
              <button type="button" className="ghost-btn mini" onClick={onResetIntrinsics}>Reset to Default</button>
              <button type="button" className="ghost-btn mini" onClick={onDismissIntrinsics}>Close</button>
            </div>
          </div>
        </div>
      )}
    </section>
  )
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------
export default function App() {
  const [pose, setPose] = React.useState(INITIAL_POSE)
  const [preset, setPreset] = React.useState('AP')
  const [drrs, setDrrs] = React.useState([])
  const [isLoading, setIsLoading] = React.useState(false)
  const [error, setError] = React.useState('')
  const [sceneInfo, setSceneInfo] = React.useState(null)
  const [interactiveMode, setInteractiveMode] = React.useState(false)
  const [extrinsicMatrix, setExtrinsicMatrix] = React.useState(null)
  const [threshold, setThreshold] = React.useState(300)
  const [intrinsicsOpen, setIntrinsicsOpen] = React.useState(false)
  const [intrinsicsData, setIntrinsicsData] = React.useState(null)
  const [targetImage, setTargetImage] = React.useState(null)
  const [overlayAlpha, setOverlayAlpha] = React.useState(0.4)
  const [selectedMetric, setSelectedMetric] = React.useState('ncc')
  const [availableMetrics, setAvailableMetrics] = React.useState(['ncc'])
  const [selectedOptimizer, setSelectedOptimizer] = React.useState('scipy_powell')
  const [isRegistering, setIsRegistering] = React.useState(false)
  const [regProgress, setRegProgress] = React.useState(null)
  const [sessionReady, setSessionReady] = React.useState(false)

  const debounceTimerRef = React.useRef(null)
  const abortControllerRef = React.useRef(null)
  const fileInputRef = React.useRef(null)
  const volumeInputRef = React.useRef(null)
  const [volumeName, setVolumeName] = React.useState(null)
  const [isUploadingVolume, setIsUploadingVolume] = React.useState(false)

  // --- WebSocket session management ---
  const sessionIdRef = React.useRef(null)
  const wsRef = React.useRef(null)

  const apiUrl = React.useCallback((path) => {
    const sep = path.includes('?') ? '&' : '?'
    return `${API_BASE}${path}${sep}session_id=${sessionIdRef.current}`
  }, [])

  React.useEffect(() => {
    let ws
    let reconnectTimer
    let disposed = false

    function connect() {
      ws = new WebSocket(`${WS_BASE}/ws`)
      wsRef.current = ws

      ws.onopen = () => {
        console.log('WebSocket connected')
      }

      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data)
        if (msg.type === 'session_start') {
          sessionIdRef.current = msg.session_id
          setSessionReady(true)
          console.log('Session started:', msg.session_id)
        } else if (msg.type === 'progress') {
          const progress = msg.data
          setRegProgress({ iteration: progress.iteration, metric_value: progress.metric_value })
          setDrrs([{ view: 'Registration', image: progress.drr }])
          setPose(progress.pose)
        } else if (msg.type === 'complete') {
          const result = msg.data
          setRegProgress({ iteration: result.iterations, metric_value: result.metric_value })
          setDrrs([{ view: 'Registration', image: result.drr }])
          setPose(result.pose)
          setIsRegistering(false)
        } else if (msg.type === 'cancelled') {
          setIsRegistering(false)
        } else if (msg.type === 'error') {
          setError(msg.data?.message || msg.message || 'Unknown error')
          setIsRegistering(false)
        }
      }

      ws.onclose = () => {
        console.log('WebSocket disconnected')
        wsRef.current = null
        sessionIdRef.current = null
        setSessionReady(false)
        if (!disposed) {
          reconnectTimer = setTimeout(connect, 2000)
        }
      }

      ws.onerror = (err) => {
        console.error('WebSocket error:', err)
        ws.close()
      }
    }

    connect()

    return () => {
      disposed = true
      clearTimeout(reconnectTimer)
      if (ws) ws.close()
    }
  }, [])

  // Fetch static scene geometry from backend once session is ready, and on preset change
  React.useEffect(() => {
    if (!sessionReady) return
    fetch(apiUrl(`/api/scene?preset=${preset}`))
      .then((r) => r.json())
      .then(setSceneInfo)
      .catch((err) => console.error('Failed to fetch scene info:', err))
  }, [preset, sessionReady, apiUrl])

  // Fetch available registration metrics on mount
  React.useEffect(() => {
    fetch(`${API_BASE}/api/registration/metrics`)
      .then((r) => r.json())
      .then((data) => setAvailableMetrics(data.metrics || ['ncc']))
      .catch((err) => console.error('Failed to fetch metrics:', err))
  }, [])

  const availablePresets = sceneInfo?.available_presets || ['AP']
  const posePayload = React.useMemo(() => ({ pose, preset, threshold }), [pose, preset, threshold])
  const hasDrrData = drrs.some((item) => item.image)
  const displayViews = drrs.length === 0 ? PLACEHOLDER_VIEWS : drrs

  const updatePose = (key, value) => {
    if (typeof value === 'function') {
      setPose((prev) => ({ ...prev, [key]: value(prev[key]) }))
    } else {
      setPose((prev) => ({ ...prev, [key]: Number(value) }))
    }
    setExtrinsicMatrix(null) // clear stale transform
  }

  const resetPose = () => {
    setPose(INITIAL_POSE)
    setError('')
    setExtrinsicMatrix(null)
  }

  const changePreset = (newPreset) => {
    setPreset(newPreset)
    setPose(INITIAL_POSE)
    setDrrs([])
    setExtrinsicMatrix(null)
    setError('')
  }

  const resetGroup = (group) => {
    const groupParams = PARAM_CONFIG.filter((p) => p.group === group)
    setPose((prev) => {
      const next = { ...prev }
      groupParams.forEach((p) => { next[p.key] = 0 })
      return next
    })
    setExtrinsicMatrix(null)
  }

  // Shared fetch logic — used by both manual button and interactive auto-generate
  const fetchDrr = React.useCallback(async (payload, signal) => {
    setIsLoading(true)
    setError('')
    try {
      const response = await fetch(apiUrl('/api/drr/generate'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal,
      })
      if (!response.ok) throw new Error(`Request failed with status ${response.status}`)
      const data = await response.json()
      setDrrs(data.drrs || [])
    } catch (err) {
      if (err.name === 'AbortError') return // superseded by newer request
      setError(err.message || 'Failed to generate DRR')
      setDrrs([])
    } finally {
      setIsLoading(false)
    }
  }, [apiUrl])

  const generateDrr = () => fetchDrr(posePayload)

  // Fetch the full 4x4 world-to-camera extrinsic matrix
  const fetchTransform = async () => {
    try {
      const response = await fetch(apiUrl('/api/camera/transform'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pose, preset }),
      })
      if (!response.ok) throw new Error(`Request failed with status ${response.status}`)
      const data = await response.json()
      setExtrinsicMatrix(data.extrinsic_4x4)
    } catch (err) {
      setError(err.message || 'Failed to fetch transform')
    }
  }

  // Fetch and open intrinsics modal
  const openIntrinsics = async () => {
    try {
      const response = await fetch(apiUrl('/api/intrinsics'))
      if (!response.ok) throw new Error(`Request failed with status ${response.status}`)
      const data = await response.json()
      setIntrinsicsData({ fx: data.fx, fy: data.fy, cx: data.cx, cy: data.cy })
      setIntrinsicsOpen(true)
    } catch (err) {
      setError(err.message || 'Failed to fetch intrinsics')
    }
  }

  const applyIntrinsics = async () => {
    if (!intrinsicsData) return
    try {
      const response = await fetch(apiUrl('/api/intrinsics'), {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(intrinsicsData),
      })
      if (!response.ok) throw new Error(`Request failed with status ${response.status}`)
      setIntrinsicsOpen(false)
      // Re-render if in interactive mode
      if (interactiveMode) {
        if (abortControllerRef.current) abortControllerRef.current.abort()
        const controller = new AbortController()
        abortControllerRef.current = controller
        fetchDrr({ pose, preset, threshold }, controller.signal)
      }
    } catch (err) {
      setError(err.message || 'Failed to apply intrinsics')
    }
  }

  const resetIntrinsics = async () => {
    try {
      const response = await fetch(apiUrl('/api/intrinsics/reset'), { method: 'POST' })
      if (!response.ok) throw new Error(`Request failed with status ${response.status}`)
      const data = await response.json()
      setIntrinsicsData({ fx: data.fx, fy: data.fy, cx: data.cx, cy: data.cy })
    } catch (err) {
      setError(err.message || 'Failed to reset intrinsics')
    }
  }

  // --- Registration handlers ---
  const handleVolumeUpload = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    setIsUploadingVolume(true)
    setError(null)
    const formData = new FormData()
    formData.append('file', file)
    try {
      const response = await fetch(apiUrl('/api/volume/upload'), {
        method: 'POST',
        body: formData,
      })
      if (!response.ok) {
        const err = await response.json().catch(() => ({}))
        throw new Error(err.detail || `Upload failed with status ${response.status}`)
      }
      const data = await response.json()
      setVolumeName(data.filename || file.name)
      setSceneInfo(data)
      setDrrs([])
      setTargetImage(null)
    } catch (err) {
      setError(err.message || 'Failed to upload volume')
    } finally {
      setIsUploadingVolume(false)
      e.target.value = ''
    }
  }

  const handleClearVolume = async () => {
    setError(null)
    try {
      const response = await fetch(apiUrl('/api/volume/clear'), { method: 'POST' })
      if (!response.ok) {
        const err = await response.json().catch(() => ({}))
        throw new Error(err.detail || 'Failed to clear volume')
      }
      setVolumeName(null)
      setSceneInfo(null)
      setDrrs([])
      setTargetImage(null)
    } catch (err) {
      setError(err.message || 'Failed to clear volume')
    }
  }

  const handleTargetUpload = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    const formData = new FormData()
    formData.append('file', file)
    try {
      const response = await fetch(apiUrl('/api/registration/target'), {
        method: 'POST',
        body: formData,
      })
      if (!response.ok) throw new Error(`Upload failed with status ${response.status}`)
      const data = await response.json()
      setTargetImage(data.target_image)
    } catch (err) {
      setError(err.message || 'Failed to upload target image')
    }
    // Reset file input so the same file can be re-uploaded
    e.target.value = ''
  }

  const clearTarget = async () => {
    try {
      await fetch(apiUrl('/api/registration/target'), { method: 'DELETE' })
      setTargetImage(null)
    } catch (err) {
      setError(err.message || 'Failed to clear target')
    }
  }

  const startRegistration = () => {
    const ws = wsRef.current
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      setError('WebSocket not connected')
      return
    }
    setIsRegistering(true)
    setRegProgress(null)
    setError('')
    ws.send(JSON.stringify({
      type: 'registration_start',
      pose, preset, threshold,
      metric: selectedMetric,
      optimizer: selectedOptimizer,
      report_every_n: 5,
    }))
  }

  const cancelRegistration = () => {
    const ws = wsRef.current
    if (!ws || ws.readyState !== WebSocket.OPEN) return
    ws.send(JSON.stringify({ type: 'registration_cancel' }))
  }

  // Interactive mode: auto-generate DRR on pose change with debounce + abort
  React.useEffect(() => {
    if (!interactiveMode) return
    if (debounceTimerRef.current) clearTimeout(debounceTimerRef.current)
    debounceTimerRef.current = setTimeout(() => {
      if (abortControllerRef.current) abortControllerRef.current.abort()
      const controller = new AbortController()
      abortControllerRef.current = controller
      fetchDrr({ pose, preset, threshold }, controller.signal)
    }, 150)
    return () => {
      if (debounceTimerRef.current) clearTimeout(debounceTimerRef.current)
    }
  }, [pose, preset, threshold, interactiveMode, fetchDrr])

  // Cleanup on unmount
  React.useEffect(() => () => {
    if (abortControllerRef.current) abortControllerRef.current.abort()
    if (debounceTimerRef.current) clearTimeout(debounceTimerRef.current)
  }, [])

  return (
    <main className="app-shell">
      <header className="topbar-card">
        <div className="topbar-head">
          <h1>DRR Workbench</h1>
        </div>
        <div className="topbar-status">
          <span className={`status-chip ${isLoading ? 'busy' : hasDrrData ? 'ok' : 'idle'}`}>
            {isUploadingVolume ? 'Loading volume…' : isLoading ? 'Rendering…' : hasDrrData ? 'DRR ready' : 'No DRR yet'}
          </span>
          {volumeName && <span className="volume-name">{volumeName}</span>}
        </div>
        <div className="topbar-actions">
          <input
            type="file"
            ref={volumeInputRef}
            accept=".nii,.nii.gz,.gz,.hdr,.img"
            style={{ display: 'none' }}
            onChange={handleVolumeUpload}
          />
          <button
            type="button"
            className="secondary-btn"
            onClick={() => volumeInputRef.current?.click()}
            disabled={isUploadingVolume || isRegistering}
          >
            {isUploadingVolume ? 'Loading…' : 'Load Volume'}
          </button>
          {volumeName && (
            <button
              type="button"
              className="secondary-btn"
              onClick={handleClearVolume}
              disabled={isUploadingVolume || isRegistering || isLoading}
            >
              Clear Volume
            </button>
          )}
          {!interactiveMode && (
            <button type="button" className="primary-btn" onClick={generateDrr} disabled={isLoading}>
              {isLoading ? 'Generating…' : 'Generate DRR'}
            </button>
          )}
          <label className="mode-toggle">
            <input
              type="checkbox"
              checked={interactiveMode}
              onChange={(e) => setInteractiveMode(e.target.checked)}
            />
            <span className="mode-toggle-track" />
            <span className="mode-toggle-label">{interactiveMode ? 'Interactive' : 'Manual'}</span>
          </label>
        </div>
      </header>

      <div className="workspace-wrap">
        {error ? <p className="error-text">{error}</p> : null}
        <section className="workspace-layout">
          <section className="results-panel">
            {/* SVG filter for red-channel overlay */}
            <svg style={{position:'absolute',width:0,height:0}}>
              <defs>
                <filter id="red-channel">
                  <feColorMatrix type="matrix" values="1 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 1 0"/>
                </filter>
              </defs>
            </svg>
            <header className="results-head">
              <h2>Projection</h2>
              <p className="results-note">
                {isRegistering
                  ? `Registering… iter ${regProgress?.iteration || 0}`
                  : interactiveMode ? 'Auto-updates on pose change.' : 'Click Generate DRR to render.'}
              </p>
            </header>
            <section className="results-grid">
              {displayViews.map((drr) => (
                <figure key={drr.view} className="drr-tile">
                  <div className="drr-viewport">
                    {drr.image ? (
                      <img src={drr.image} alt={drr.view} />
                    ) : (
                      <div className={`drr-empty ${isLoading ? 'loading' : ''}`}>
                        {isLoading ? 'Rendering…' : interactiveMode ? 'Adjust pose to generate' : 'Click Generate DRR'}
                      </div>
                    )}
                    {targetImage && (
                      <img
                        src={targetImage}
                        alt="Target overlay"
                        className="target-overlay"
                        style={{ opacity: overlayAlpha }}
                      />
                    )}
                    {isLoading && drr.image ? <div className="tile-overlay">Updating…</div> : null}
                  </div>
                  <figcaption>
                    <span className="view-tag">{drr.image ? 'Ready' : 'Pending'}</span>
                  </figcaption>
                </figure>
              ))}
            </section>
            {targetImage && (
              <div className="overlay-controls">
                <label className="overlay-alpha-label">Overlay</label>
                <input
                  type="range"
                  min={0} max={1} step={0.05}
                  value={overlayAlpha}
                  onChange={(e) => setOverlayAlpha(Number(e.target.value))}
                  className="overlay-alpha-slider"
                />
                <span className="overlay-alpha-value">{Math.round(overlayAlpha * 100)}%</span>
              </div>
            )}
          </section>

          <div className="right-col">
            <FrameIllustration pose={pose} sceneInfo={sceneInfo} />
            <section className="threshold-panel">
              <label className="threshold-label">Threshold (HU)</label>
              <input
                className="threshold-input"
                type="number"
                step={50}
                value={threshold}
                onChange={(e) => {
                  const v = Number(e.target.value)
                  if (!Number.isNaN(v)) setThreshold(v)
                }}
              />
            </section>
            <section className="registration-panel">
              <h3 className="reg-panel-title">Registration</h3>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/png,image/jpeg,image/bmp"
                style={{ display: 'none' }}
                onChange={handleTargetUpload}
              />
              <div className="reg-target-row">
                {targetImage ? (
                  <>
                    <span className="reg-target-status">Target loaded</span>
                    <button type="button" className="ghost-btn tiny" onClick={clearTarget}>Clear</button>
                  </>
                ) : (
                  <button
                    type="button"
                    className="ghost-btn mini"
                    onClick={() => fileInputRef.current?.click()}
                  >Upload Target X-ray</button>
                )}
              </div>
              <div className="reg-metric-row">
                <label className="reg-metric-label">Optimizer</label>
                <select
                  className="preset-select"
                  value={selectedOptimizer}
                  onChange={(e) => setSelectedOptimizer(e.target.value)}
                >
                  <option value="scipy_powell">Scipy - Powell</option>
                  <option value="pytorch_adamw">PyTorch - AdamW</option>
                  <option value="pytorch_sgd">PyTorch - SGD</option>
                </select>
              </div>
              <div className="reg-metric-row">
                <label className="reg-metric-label">Metric</label>
                <select
                  className="preset-select"
                  value={selectedMetric}
                  onChange={(e) => setSelectedMetric(e.target.value)}
                >
                  {availableMetrics.map((m) => (
                    <option key={m} value={m}>{m}</option>
                  ))}
                </select>
              </div>
              <div className="reg-btn-group">
                {isRegistering ? (
                  <button type="button" className="ghost-btn mini reg-cancel-btn" onClick={cancelRegistration}>Cancel</button>
                ) : (
                  <button
                    type="button"
                    className="primary-btn mini"
                    disabled={!targetImage}
                    onClick={startRegistration}
                  >Start Registration</button>
                )}
              </div>
              {regProgress && (
                <div className="reg-progress">
                  <span>Iter: {regProgress.iteration}</span>
                  <span>Metric: {regProgress.metric_value?.toFixed(4)}</span>
                </div>
              )}
            </section>
            <PoseControlsPanel
              pose={pose}
              onChangePose={updatePose}
              onResetGroup={resetGroup}
              onResetPose={resetPose}
              preset={preset}
              availablePresets={availablePresets}
              onChangePreset={changePreset}
              onFetchTransform={fetchTransform}
              extrinsicMatrix={extrinsicMatrix}
              onDismissTransform={() => setExtrinsicMatrix(null)}
              onOpenIntrinsics={openIntrinsics}
              intrinsicsOpen={intrinsicsOpen}
              intrinsicsData={intrinsicsData}
              onIntrinsicsChange={setIntrinsicsData}
              onApplyIntrinsics={applyIntrinsics}
              onResetIntrinsics={resetIntrinsics}
              onDismissIntrinsics={() => setIntrinsicsOpen(false)}
            />
          </div>
        </section>
      </div>
    </main>
  )
}
