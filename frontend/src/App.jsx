import * as React from 'react'
import { Canvas } from '@react-three/fiber'
import { Line, OrbitControls, Text } from '@react-three/drei'
import * as THREE from 'three'

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

const PARAM_CONFIG = [
  { key: 'tx', label: 'Tx', min: -200, max: 200, step: 1, unit: 'mm', group: 'translation' },
  { key: 'ty', label: 'Ty', min: -200, max: 200, step: 1, unit: 'mm', group: 'translation' },
  { key: 'tz', label: 'Tz', min: -200, max: 200, step: 1, unit: 'mm', group: 'translation' },
  { key: 'rx', label: 'Rx', min: -180, max: 180, step: 1, unit: 'deg', group: 'rotation' },
  { key: 'ry', label: 'Ry', min: -180, max: 180, step: 1, unit: 'deg', group: 'rotation' },
  { key: 'rz', label: 'Rz', min: -180, max: 180, step: 1, unit: 'deg', group: 'rotation' },
]

const INITIAL_POSE = {
  tx: 0,
  ty: 0,
  tz: 0,
  rx: 0,
  ry: 0,
  rz: 0,
}

const AXIS_COLORS = {
  x: '#ff5a4f',
  y: '#4caf50',
  z: '#2f7de1',
}

const PLACEHOLDER_VIEWS = [{ view: 1 }, { view: 2 }, { view: 3 }, { view: 4 }]

const MM_TO_SCENE = 0.115
const FRAME_CAMERA_DEFAULT = {
  position: [7.2, 2.4, 7.1],
  target: [0, -0.1, -0.2],
  fov: 34,
}

const toRad = (deg) => (deg * Math.PI) / 180

function rotationFromEuler({ rx, ry, rz }) {
  const ax = toRad(rx)
  const ay = toRad(ry)
  const az = toRad(rz)

  const sx = Math.sin(ax)
  const cx = Math.cos(ax)
  const sy = Math.sin(ay)
  const cy = Math.cos(ay)
  const sz = Math.sin(az)
  const cz = Math.cos(az)

  // Project convention: R = Rz * Ry * Rx
  return [
    [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
    [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
    [-sy, cy * sx, cy * cx],
  ]
}

const transpose3 = (m) => [
  [m[0][0], m[1][0], m[2][0]],
  [m[0][1], m[1][1], m[2][1]],
  [m[0][2], m[1][2], m[2][2]],
]

const mulMatVec = (m, v) => ({
  x: m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
  y: m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
  z: m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z,
})

const subVec = (a, b) => ({ x: a.x - b.x, y: a.y - b.y, z: a.z - b.z })

const clamp = (value, min, max) => Math.max(min, Math.min(max, value))

const toScenePoint = (v) => [v.x * MM_TO_SCENE, v.z * MM_TO_SCENE, -v.y * MM_TO_SCENE]
const toSceneVec = (v) => new THREE.Vector3(v.x, v.z, -v.y)

function basisToQuaternion(basis) {
  const bx = toSceneVec(basis.x).normalize()
  const by = toSceneVec(basis.y).normalize()
  const bz = toSceneVec(basis.z).normalize()

  const matrix = new THREE.Matrix4().set(
    bx.x, by.x, bz.x, 0,
    bx.y, by.y, bz.y, 0,
    bx.z, by.z, bz.z, 0,
    0, 0, 0, 1,
  )

  return new THREE.Quaternion().setFromRotationMatrix(matrix)
}

function AxisTriad({ origin, basis, length = 3.1, dashed = false, opacity = 1 }) {
  const axes = React.useMemo(() => {
    const start = new THREE.Vector3(...toScenePoint(origin))
    const up = new THREE.Vector3(0, 1, 0)

    return [
      { key: 'x', color: AXIS_COLORS.x, dir: toSceneVec(basis.x).normalize() },
      { key: 'y', color: AXIS_COLORS.y, dir: toSceneVec(basis.y).normalize() },
      { key: 'z', color: AXIS_COLORS.z, dir: toSceneVec(basis.z).normalize() },
    ].map((entry) => {
      const end = start.clone().addScaledVector(entry.dir, length)
      const tipQ = new THREE.Quaternion().setFromUnitVectors(up, entry.dir)

      return {
        ...entry,
        start,
        end,
        tipPos: end.toArray(),
        tipQ,
      }
    })
  }, [origin, basis, length])

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
        </group>
      ))}

      <mesh position={toScenePoint(origin)}>
        <sphereGeometry args={[0.13, 20, 20]} />
        <meshStandardMaterial color="#d88b00" />
      </mesh>
    </group>
  )
}

function PatientModel({ origin }) {
  const position = React.useMemo(() => toScenePoint(origin), [origin])

  return (
    <group position={position}>
      <mesh position={[0, 1.95, 0]}>
        <sphereGeometry args={[0.46, 24, 24]} />
        <meshStandardMaterial color="#1f49e0" metalness={0.08} roughness={0.3} />
      </mesh>

      <mesh position={[0, 0.78, 0]}>
        <cylinderGeometry args={[0.52, 0.64, 2.15, 22]} />
        <meshStandardMaterial color="#1944de" metalness={0.06} roughness={0.35} />
      </mesh>

      <mesh position={[-0.8, 1.02, 0]} rotation={[0, 0, Math.PI / 2.6]}>
        <cylinderGeometry args={[0.11, 0.11, 1.15, 16]} />
        <meshStandardMaterial color="#0d2ea6" />
      </mesh>
      <mesh position={[0.8, 1.02, 0]} rotation={[0, 0, -Math.PI / 2.6]}>
        <cylinderGeometry args={[0.11, 0.11, 1.15, 16]} />
        <meshStandardMaterial color="#0d2ea6" />
      </mesh>

      <mesh position={[-0.28, -0.6, 0]} rotation={[0, 0, 0.14]}>
        <cylinderGeometry args={[0.12, 0.12, 1.62, 16]} />
        <meshStandardMaterial color="#0d2ea6" />
      </mesh>
      <mesh position={[0.28, -0.6, 0]} rotation={[0, 0, -0.14]}>
        <cylinderGeometry args={[0.12, 0.12, 1.62, 16]} />
        <meshStandardMaterial color="#0d2ea6" />
      </mesh>
    </group>
  )
}

function CTVolume({ origin, cubeHalfMm }) {
  const pos = React.useMemo(() => toScenePoint(origin), [origin])
  const cubeHalf = cubeHalfMm * MM_TO_SCENE
  const side = cubeHalf * 2

  const { corners, edges } = React.useMemo(() => {
    const c = [
      [-cubeHalf, -cubeHalf, -cubeHalf],
      [cubeHalf, -cubeHalf, -cubeHalf],
      [cubeHalf, cubeHalf, -cubeHalf],
      [-cubeHalf, cubeHalf, -cubeHalf],
      [-cubeHalf, -cubeHalf, cubeHalf],
      [cubeHalf, -cubeHalf, cubeHalf],
      [cubeHalf, cubeHalf, cubeHalf],
      [-cubeHalf, cubeHalf, cubeHalf],
    ]

    return {
      corners: c,
      edges: [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
      ],
    }
  }, [cubeHalf])

  return (
    <group position={pos}>
      <mesh>
        <boxGeometry args={[side, side, side]} />
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
        position={[cubeHalf + 0.62, -cubeHalf * 0.48, cubeHalf * 0.74]}
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
  const position = React.useMemo(() => toScenePoint(origin), [origin])
  const quaternion = React.useMemo(() => basisToQuaternion(basis), [basis])

  const frustum = React.useMemo(() => {
    const near = 1.8
    const hw = 0.82
    const hh = 0.56

    return [
      [-hw, -hh, near],
      [hw, -hh, near],
      [hw, hh, near],
      [-hw, hh, near],
    ]
  }, [])

  return (
    <group position={position} quaternion={quaternion} scale={[0.74, 0.74, 0.74]}>
      <mesh position={[0, 0, 0]}>
        <boxGeometry args={[0.92, 0.58, 0.62]} />
        <meshStandardMaterial color="#2f80df" metalness={0.1} roughness={0.35} />
      </mesh>

      <mesh position={[0, 0, 0.45]} rotation={[Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[0.18, 0.18, 0.24, 20]} />
        <meshStandardMaterial color="#f5f9ff" metalness={0.04} roughness={0.28} />
      </mesh>

      <mesh position={[0, 0.28, 0.04]}>
        <boxGeometry args={[0.32, 0.09, 0.22]} />
        <meshStandardMaterial color="#1b58a0" />
      </mesh>

      <mesh position={[0, 0, 1.8]}>
        <planeGeometry args={[1.64, 1.12]} />
        <meshStandardMaterial color="#4f9cf1" transparent opacity={0.2} side={THREE.DoubleSide} />
      </mesh>

      <Line points={[frustum[0], frustum[1], frustum[2], frustum[3], frustum[0]]} color="#4d93e2" lineWidth={1.2} />
      {frustum.map((corner, idx) => (
        <Line key={`frustum-ray-${idx}`} points={[[0, 0, 0.12], corner]} color="#4d93e2" lineWidth={1.1} />
      ))}
    </group>
  )
}

function FrameScene({ pose }) {
  const sceneState = React.useMemo(() => {
    const translationScale = 0.22
    const poseTranslation = {
      x: clamp(pose.tx * translationScale, -75, 75),
      y: clamp(pose.ty * translationScale, -75, 75),
      z: clamp(pose.tz * translationScale, -75, 75),
    }

    const rotation = rotationFromEuler(pose)
    const invRotation = transpose3(rotation)

    const volumeOrigin = { x: 0, y: 0, z: -8 }
    const patientOrigin = { x: 0, y: 0, z: -8 }

    // Hard-coded AP camera extrinsic (camera->world).
    const apCameraExtrinsic = {
      translation: { x: 0, y: 120, z: 16 },
      basis: {
        x: { x: 1, y: 0, z: 0 },
        y: { x: 0, y: 0, z: 1 },
        z: { x: 0, y: -1, z: 0 },
      },
    }

    // Sliders apply CT transform; for this widget CT is fixed and camera is moved by inverse transform.
    const cameraOrigin = mulMatVec(invRotation, subVec(apCameraExtrinsic.translation, poseTranslation))
    const cameraBasis = {
      x: mulMatVec(invRotation, apCameraExtrinsic.basis.x),
      y: mulMatVec(invRotation, apCameraExtrinsic.basis.y),
      z: mulMatVec(invRotation, apCameraExtrinsic.basis.z),
    }

    const patientBasis = {
      x: { x: 1, y: 0, z: 0 },
      y: { x: 0, y: 1, z: 0 },
      z: { x: 0, y: 0, z: 1 },
    }

    const volumeBasis = patientBasis

    return {
      patientOrigin,
      patientBasis,
      volumeOrigin,
      volumeBasis,
      cameraOrigin,
      cameraBasis,
      ctHalfMm: 28 * 0.72,
    }
  }, [pose])

  return (
    <>
      <ambientLight intensity={0.78} />
      <directionalLight position={[16, 22, 12]} intensity={1.1} />
      <directionalLight position={[-12, 9, -16]} intensity={0.45} />

      <CTVolume origin={sceneState.volumeOrigin} cubeHalfMm={sceneState.ctHalfMm} />
      <PatientModel origin={sceneState.patientOrigin} />
      <APCameraModel origin={sceneState.cameraOrigin} basis={sceneState.cameraBasis} />

      <AxisTriad origin={sceneState.patientOrigin} basis={sceneState.patientBasis} length={3.4} opacity={0.95} />
      <AxisTriad origin={sceneState.cameraOrigin} basis={sceneState.cameraBasis} length={2.4} opacity={0.9} />
      <AxisTriad origin={sceneState.volumeOrigin} basis={sceneState.volumeBasis} length={2.65} opacity={0.82} />
    </>
  )
}

function FrameIllustration({ pose }) {
  const controlsRef = React.useRef(null)

  const resetView = () => {
    const controls = controlsRef.current
    if (!controls) return

    controls.object.position.set(...FRAME_CAMERA_DEFAULT.position)
    controls.target.set(...FRAME_CAMERA_DEFAULT.target)
    controls.update()
  }

  return (
    <section className="frame-panel">
      <div className="panel-title-row">
        <div className="panel-heading">
          <h2>Frame Sketch</h2>
          <p>Orbit, pan, and zoom</p>
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
          <FrameScene pose={pose} />
          <OrbitControls
            ref={controlsRef}
            makeDefault
            enableDamping
            dampingFactor={0.12}
            rotateSpeed={0.9}
            panSpeed={0.7}
            minDistance={4}
            maxDistance={22}
            target={FRAME_CAMERA_DEFAULT.target}
          />
        </Canvas>
      </div>
    </section>
  )
}

function PoseAxisCard({ param, value, onChange }) {
  const quickStep = param.group === 'translation' ? 5 : 2
  const coarseStep = param.group === 'translation' ? 20 : 10

  const handleNumber = (event) => {
    const next = Number(event.target.value)
    if (Number.isNaN(next)) return
    onChange(param.key, clamp(next, param.min, param.max))
  }

  const applyDelta = (delta) => {
    onChange(param.key, clamp(value + delta, param.min, param.max))
  }

  return (
    <article className={`pose-axis-card ${param.group}`}>
      <header className="pose-axis-card-head">
        <div className="pose-axis-meta">
          <span className={`pose-axis-dot ${param.group}`} />
          <span className="pose-axis-label">{param.label}</span>
        </div>
        <span className="pose-axis-unit">{param.unit}</span>
      </header>

      <div className="pose-axis-value-wrap">
        <button
          type="button"
          className="axis-step-btn"
          onClick={() => applyDelta(-quickStep)}
          aria-label={`Decrease ${param.label} by ${quickStep}`}
        >
          -{quickStep}
        </button>

        <input
          className="pose-axis-value"
          type="number"
          min={param.min}
          max={param.max}
          step={param.step}
          value={value}
          onChange={handleNumber}
          aria-label={`${param.label} value`}
        />

        <button
          type="button"
          className="axis-step-btn"
          onClick={() => applyDelta(quickStep)}
          aria-label={`Increase ${param.label} by ${quickStep}`}
        >
          +{quickStep}
        </button>
      </div>

      <div className="pose-axis-actions">
        <button
          type="button"
          className="axis-mini-btn subtle"
          onClick={() => applyDelta(-coarseStep)}
          aria-label={`Decrease ${param.label} by ${coarseStep}`}
        >
          -{coarseStep}
        </button>
        <button type="button" className="axis-zero-btn" onClick={() => onChange(param.key, 0)} aria-label={`Zero ${param.label}`}>
          0
        </button>
        <button
          type="button"
          className="axis-mini-btn"
          onClick={() => applyDelta(coarseStep)}
          aria-label={`Increase ${param.label} by ${coarseStep}`}
        >
          +{coarseStep}
        </button>
      </div>

      <input
        className={`axis-range card ${param.group}`}
        type="range"
        min={param.min}
        max={param.max}
        step={param.step}
        value={value}
        onChange={(event) => onChange(param.key, Number(event.target.value))}
        aria-label={`${param.label} slider`}
      />
    </article>
  )
}

function PoseControlsPanel({
  pose,
  onChangePose,
  onResetGroup,
  onResetPose,
}) {
  const translationControls = PARAM_CONFIG.filter((param) => param.group === 'translation')
  const rotationControls = PARAM_CONFIG.filter((param) => param.group === 'rotation')

  return (
    <section className="pose-panel">
      <header className="pose-panel-head">
        <div>
          <p className="eyebrow">Pose Control</p>
          <h2>Camera Transform</h2>
        </div>
        <button type="button" className="ghost-btn tiny" onClick={onResetPose}>
          Reset Pose
        </button>
      </header>

      <div className="pose-sections">
        <section className="axis-group">
          <header className="axis-group-head">
            <h3>Translation (mm)</h3>
            <button type="button" className="ghost-btn tiny" onClick={() => onResetGroup('translation')}>
              Zero T
            </button>
          </header>
          <div className="axis-card-grid">
            {translationControls.map((param) => (
              <PoseAxisCard key={param.key} param={param} value={pose[param.key]} onChange={onChangePose} />
            ))}
          </div>
        </section>

        <section className="axis-group">
          <header className="axis-group-head">
            <h3>Rotation (deg)</h3>
            <button type="button" className="ghost-btn tiny" onClick={() => onResetGroup('rotation')}>
              Zero R
            </button>
          </header>
          <div className="axis-card-grid">
            {rotationControls.map((param) => (
              <PoseAxisCard key={param.key} param={param} value={pose[param.key]} onChange={onChangePose} />
            ))}
          </div>
        </section>
      </div>

      <p className="pose-convention">R = Rz * Ry * Rx, p_out = R*p_in + t.</p>
    </section>
  )
}

export default function App() {
  const [pose, setPose] = React.useState(INITIAL_POSE)
  const [drrs, setDrrs] = React.useState([])
  const [isLoading, setIsLoading] = React.useState(false)
  const [error, setError] = React.useState('')

  const posePayload = React.useMemo(() => ({ pose }), [pose])
  const hasDrrData = drrs.some((item) => item.image)
  const displayViews = drrs.length === 0 ? PLACEHOLDER_VIEWS : drrs

  const updatePose = (key, value) => {
    setPose((prev) => ({ ...prev, [key]: Number(value) }))
  }

  const resetPose = () => {
    setPose(INITIAL_POSE)
    setError('')
  }

  const resetGroup = (group) => {
    const groupParams = PARAM_CONFIG.filter((param) => param.group === group)
    setPose((prev) => {
      const next = { ...prev }
      groupParams.forEach((param) => {
        next[param.key] = 0
      })
      return next
    })
  }

  const generateDrr = async () => {
    setIsLoading(true)
    setError('')

    try {
      const response = await fetch(`${API_BASE}/api/drr/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(posePayload),
      })

      if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}`)
      }

      const data = await response.json()
      setDrrs(data.drrs || [])
    } catch (err) {
      setError(err.message || 'Failed to generate DRR')
      setDrrs([])
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar-card">
        <div className="topbar-head">
          <p className="eyebrow">2D/3D Registration</p>
          <h1>AP Camera Workspace</h1>
          <p className="subtitle">Keep CT fixed, tune camera pose, and regenerate DRR projections.</p>
        </div>

        <div className="topbar-status">
          <span className={`status-chip ${isLoading ? 'busy' : hasDrrData ? 'ok' : 'idle'}`}>
            {isLoading ? 'Rendering...' : hasDrrData ? 'Views ready' : 'No DRR yet'}
          </span>
          <p className="status-note">Frame widget updates live with each axis change.</p>
        </div>

        <div className="topbar-actions">
          <button type="button" className="primary-btn" onClick={generateDrr} disabled={isLoading}>
            {isLoading ? 'Generating DRR...' : 'Generate DRR'}
          </button>
          <button type="button" className="ghost-btn" onClick={resetPose}>
            Reset Pose
          </button>
        </div>
      </header>

      {error ? <p className="error-text">{error}</p> : null}

      <section className="workspace-layout">
        <section className="results-panel">
          <header className="results-head">
            <div>
              <p className="eyebrow">Output</p>
              <h2>DRR Views</h2>
            </div>
            <p className="results-note">AP camera returns 4 projections from the backend stub.</p>
          </header>

          <section className="results-grid">
            {displayViews.map((drr) => (
              <figure key={drr.view} className="drr-tile">
                <div className="drr-viewport">
                  {drr.image ? (
                    <img src={drr.image} alt={`DRR view ${drr.view}`} />
                  ) : (
                    <div className={`drr-empty ${isLoading ? 'loading' : ''}`}>
                      {isLoading ? 'Rendering projection...' : 'Generate to load image'}
                    </div>
                  )}
                  {isLoading && drr.image ? <div className="tile-overlay">Updating...</div> : null}
                </div>
                <figcaption>
                  <span>View {drr.view}</span>
                  <span className="view-tag">{drr.image ? 'Ready' : 'Pending'}</span>
                </figcaption>
              </figure>
            ))}
          </section>
        </section>

        <aside className="right-rail">
          <FrameIllustration pose={pose} />
          <PoseControlsPanel
            pose={pose}
            onChangePose={updatePose}
            onResetGroup={resetGroup}
            onResetPose={resetPose}
          />
        </aside>
      </section>
    </main>
  )
}
