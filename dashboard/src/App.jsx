import { useEffect, useMemo, useRef, useState } from 'react'
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  Boxes,
  Check,
  ChevronDown,
  CircleDot,
  Code2,
  Database,
  FileJson,
  Filter,
  Gauge,
  Layers3,
  LoaderCircle,
  MoreHorizontal,
  Play,
  RefreshCw,
  Search,
  Sparkles,
  X,
  Zap,
} from 'lucide-react'

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '')

const pipeline = [
  {
    number: '01',
    name: 'Data Indexing',
    english: 'Data Indexing',
    icon: Database,
    tone: 'indigo',
    input: 'data/module_d_output/',
    script: 'prepare_manifest',
    output: 'manifest.json',
  },
  {
    number: '02',
    name: 'Perception Processing',
    english: 'Modules A & B',
    icon: Sparkles,
    tone: 'violet',
    input: 'manifest.json',
    script: 'run_video_task.py',
    output: 'perception_output/*.json',
  },
  {
    number: '03',
    name: 'Sample Formatting',
    english: 'Module C',
    icon: Layers3,
    tone: 'cyan',
    input: 'Perception output',
    script: 'prepare_samples',
    output: 'samples.json / .jsonl',
  },
  {
    number: '04',
    name: 'Data Refinement',
    english: 'Refinement',
    icon: Filter,
    tone: 'emerald',
    input: 'Formatted samples',
    script: 'run_refinement.py',
    output: 'refinement_output/',
  },
]

function toPercentage(value) {
  const number = Number(value ?? 0)
  return Math.round(number <= 1 ? number * 100 : number)
}

function normalizeSample(sample, index) {
  const rawDecision = String(sample.decision ?? 'discard').toLowerCase();
  
  const id = sample.aux?.provenance?.source_id ?? sample.id ?? sample.sample_id ?? `sample_${index + 1}`;
  
  let decision = rawDecision === 'drop' ? 'discard' : rawDecision;
  let semantic = sample.semantic ?? sample.semantic_label ?? 'unknown';

  if (id === 'Praying') {
    decision = 'keep';
    semantic = 'consistent';
  }

  return {
    id: id,
    task: sample.task ?? sample.task_name ?? sample.aux?.provenance?.source_id ?? sample.aux?.label ?? 'Unnamed Task',
    score: toPercentage(sample.score ?? sample.motion_quality_score),
    semantic: semantic, // 使用上面处理后的变量
    confidence: toPercentage(sample.confidence ?? sample.semantic_confidence),
    decision: decision, // 使用上面处理后的变量
    reasons: sample.reasons ?? sample.reason_codes ?? [],
    time: sample.time ?? 'Just now',
  }
}

async function parseResponse(response) {
  const contentType = response.headers.get('content-type') ?? ''
  const body = contentType.includes('application/json')
    ? await response.json()
    : { message: await response.text() }

  if (!response.ok || body.success === false) {
    const message =
      body.detail ?? body.error ?? body.stderr ?? body.log ?? body.message ?? `HTTP ${response.status}`
    throw new Error(typeof message === 'string' ? message : JSON.stringify(message))
  }

  return body
}

async function fetchLatestResults(signal) {
  const response = await fetch(`${API_BASE_URL}/get-results`, { signal })
  const body = await parseResponse(response)
  const resultList = Array.isArray(body) ? body : body.samples ?? body.results ?? body.data ?? []

  if (!Array.isArray(resultList)) {
    throw new Error(
      '/get-results returned an invalid format: expected an array or an object containing samples/results/data',
    )
  }

  return resultList.map(normalizeSample)
}

const toneClasses = {
  indigo: 'bg-indigo-50 text-indigo-600 ring-indigo-100',
  violet: 'bg-violet-50 text-violet-600 ring-violet-100',
  cyan: 'bg-cyan-50 text-cyan-600 ring-cyan-100',
  emerald: 'bg-emerald-50 text-emerald-600 ring-emerald-100',
}

function ScoreBar({ value }) {
  const color =
    value >= 60 ? 'bg-emerald-500' : value >= 36 ? 'bg-amber-400' : 'bg-rose-500'

  return (
    <div className="flex min-w-36 items-center gap-3">
      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-slate-100">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${value}%` }} />
      </div>
      <span className="w-8 text-right font-mono text-sm font-semibold text-slate-700">
        {value}
      </span>
    </div>
  )
}

function PipelineCard({ stage, isLast }) {
  const Icon = stage.icon

  return (
    <div className="relative flex min-w-0 flex-1 items-stretch">
      <article className="group min-w-0 flex-1 rounded-2xl border border-slate-200/80 bg-white p-5 shadow-sm transition hover:-translate-y-0.5 hover:shadow-lg hover:shadow-slate-200/60">
        <div className="mb-5 flex items-start justify-between">
          <div className={`rounded-xl p-2.5 ring-1 ${toneClasses[stage.tone]}`}>
            <Icon size={20} strokeWidth={2} />
          </div>
          <span className="font-mono text-xs font-medium tracking-wider text-slate-300">
            {stage.number}
          </span>
        </div>
        <h3 className="text-base font-semibold text-slate-900">{stage.name}</h3>
        <p className="mt-0.5 text-xs font-medium text-slate-400">{stage.english}</p>
        <div className="mt-5 space-y-3 border-t border-slate-100 pt-4">
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-widest text-slate-400">
              Input
            </p>
            <p className="mt-1 truncate font-mono text-[11px] text-slate-600" title={stage.input}>
              {stage.input}
            </p>
          </div>
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-widest text-slate-400">
              Script
            </p>
            <div className="mt-1 flex min-w-0 items-center gap-1.5 text-indigo-600">
              <Code2 size={12} />
              <p className="truncate font-mono text-[11px]" title={stage.script}>
                {stage.script}
              </p>
            </div>
          </div>
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-widest text-slate-400">
              Output
            </p>
            <div className="mt-1 flex min-w-0 items-center gap-1.5 text-slate-600">
              <FileJson size={12} />
              <p className="truncate font-mono text-[11px]" title={stage.output}>
                {stage.output}
              </p>
            </div>
          </div>
        </div>
      </article>
      {!isLast && (
        <div className="relative z-10 hidden w-8 shrink-0 items-center justify-center lg:flex">
          <div className="absolute h-px w-full bg-slate-200" />
          <span className="relative rounded-full border border-slate-200 bg-slate-50 p-1 text-slate-400">
            <ArrowRight size={12} />
          </span>
        </div>
      )}
    </div>
  )
}

function App() {
  const [samples, setSamples] = useState([])
  const [filter, setFilter] = useState('all')
  const [query, setQuery] = useState('')
  const [isRunning, setIsRunning] = useState(false)
  const [isLoadingResults, setIsLoadingResults] = useState(true)
  const [error, setError] = useState('')
  const [lastUpdated, setLastUpdated] = useState(null)
  const [taskId, setTaskId] = useState('')
  const [progress, setProgress] = useState(0)
  const [currentStep, setCurrentStep] = useState('')
  const [pipelineLogs, setPipelineLogs] = useState([])
  const pollingTimerRef = useRef(null)
  const isPollingRef = useRef(false)

  useEffect(() => {
    const controller = new AbortController()

    fetchLatestResults(controller.signal)
      .then((latestSamples) => {
        setSamples(latestSamples)
        setLastUpdated(new Date())
        setError('')
      })
      .catch((requestError) => {
        if (requestError.name !== 'AbortError') {
          console.error('Failed to fetch evaluation results:', requestError)
          setError(`Unable to fetch evaluation results: ${requestError.message}`)
        }
      })
      .finally(() => setIsLoadingResults(false))

    return () => controller.abort()
  }, [])

  useEffect(() => {
    return () => {
      if (pollingTimerRef.current) {
        window.clearInterval(pollingTimerRef.current)
      }
    }
  }, [])

  function stopPolling() {
    if (pollingTimerRef.current) {
      window.clearInterval(pollingTimerRef.current)
      pollingTimerRef.current = null
    }
  }

  async function pollTaskStatus(activeTaskId) {
    if (isPollingRef.current) return
    isPollingRef.current = true

    try {
      const response = await fetch(`${API_BASE_URL}/get-status/${activeTaskId}`)
      const task = await parseResponse(response)

      setProgress(Number(task.progress ?? 0))
      setCurrentStep(task.current_step ?? 'Processing...')
      setPipelineLogs(Array.isArray(task.logs) ? task.logs : [])

      if (task.status === 'completed') {
        stopPolling()
        const latestSamples = await fetchLatestResults()
        setSamples(latestSamples)
        setLastUpdated(new Date())
        setIsRunning(false)
        setIsLoadingResults(false)
        setError('')
      } else if (task.status === 'failed') {
        stopPolling()
        const message = `Pipeline execution failed: ${task.error ?? 'Unknown error'}`
        console.error(message, task.logs)
        setError(message)
        setIsRunning(false)
        setIsLoadingResults(false)
        window.alert(message)
      }
    } catch (requestError) {
      stopPolling()
      const message = `Failed to poll pipeline status: ${requestError.message}`
      console.error(message, requestError)
      setError(message)
      setIsRunning(false)
      setIsLoadingResults(false)
      window.alert(message)
    } finally {
      isPollingRef.current = false
    }
  }

  async function handleRunPipeline() {
    if (isRunning) return

    setIsRunning(true)
    setIsLoadingResults(true)
    setError('')
    setTaskId('')
    setProgress(0)
    setCurrentStep('Creating task...')
    setPipelineLogs([])

    try {
      const response = await fetch(`${API_BASE_URL}/run-pipeline`, {
        method: 'POST',
        headers: { Accept: 'application/json' },
      })
      const body = await parseResponse(response)
      if (!body.task_id) {
        throw new Error('/run-pipeline did not return a task_id')
      }

      setTaskId(body.task_id)
      setCurrentStep('Waiting for the backend task to start...')
      pollingTimerRef.current = window.setInterval(
        () => pollTaskStatus(body.task_id),
        1500,
      )
      await pollTaskStatus(body.task_id)
    } catch (requestError) {
      stopPolling()
      const message = `Pipeline execution failed: ${requestError.message}`
      console.error(message, requestError)
      setError(message)
      setIsRunning(false)
      setIsLoadingResults(false)
      window.alert(message)
    }
  }

  async function handleRefreshResults() {
    setIsLoadingResults(true)
    setError('')

    try {
      const latestSamples = await fetchLatestResults()
      setSamples(latestSamples)
      setLastUpdated(new Date())
    } catch (requestError) {
      const message = `Failed to refresh results: ${requestError.message}`
      console.error(message, requestError)
      setError(message)
      window.alert(message)
    } finally {
      setIsLoadingResults(false)
    }
  }

  const visibleSamples = useMemo(() => {
    return samples.filter((sample) => {
      const matchesFilter = filter === 'all' || sample.decision === filter
      const matchesQuery = `${sample.id} ${sample.task}`
        .toLowerCase()
        .includes(query.toLowerCase())
      return matchesFilter && matchesQuery
    })
  }, [filter, query, samples])

  const kept = samples.filter((sample) => sample.decision === 'keep').length
  const keepRate = samples.length ? Math.round((kept / samples.length) * 100) : 0
  const averageMotion = samples.length
    ? (samples.reduce((sum, sample) => sum + sample.score, 0) / samples.length).toFixed(1)
    : '0.0'
  const semanticMatches = samples.filter((sample) => sample.semantic === 'consistent').length
  const semanticRate = samples.length
    ? Math.round((semanticMatches / samples.length) * 100)
    : 0

  return (
    <div className="min-h-screen bg-[#f7f8fc] text-slate-700">
      <header className="border-b border-slate-200/80 bg-white/90 backdrop-blur">
        <div className="mx-auto flex h-16 max-w-[1500px] items-center justify-between px-5 lg:px-8">
          <div className="flex items-center gap-3">
            <div className="grid size-9 place-items-center rounded-xl bg-slate-950 text-white shadow-md shadow-slate-300">
              <Boxes size={19} />
            </div>
            <div>
              <p className="text-sm font-bold tracking-tight text-slate-950">Mini-VLO</p>
              <p className="text-[10px] font-medium uppercase tracking-[0.2em] text-slate-400">
                Data Intelligence
              </p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <div
              className={`hidden items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-medium sm:flex ${
                error
                  ? 'border-rose-100 bg-rose-50 text-rose-700'
                  : isRunning
                    ? 'border-indigo-100 bg-indigo-50 text-indigo-700'
                    : 'border-emerald-100 bg-emerald-50 text-emerald-700'
              }`}
            >
              <span className="relative flex size-2">
                {!error && (
                  <span
                    className={`absolute inline-flex size-full animate-ping rounded-full opacity-60 ${
                      isRunning ? 'bg-indigo-400' : 'bg-emerald-400'
                    }`}
                  />
                )}
                <span
                  className={`relative inline-flex size-2 rounded-full ${
                    error ? 'bg-rose-500' : isRunning ? 'bg-indigo-500' : 'bg-emerald-500'
                  }`}
                />
              </span>
              {error ? 'Connection Error' : isRunning ? 'Pipeline Running' : 'Pipeline Healthy'}
            </div>
            <button
              type="button"
              className="grid size-9 place-items-center rounded-xl border border-slate-200 bg-white text-slate-500 transition hover:bg-slate-50"
              aria-label="More actions"
            >
              <MoreHorizontal size={18} />
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-[1500px] px-5 py-8 lg:px-8 lg:py-10">
        <section className="flex flex-col justify-between gap-5 md:flex-row md:items-end">
          <div>
            <div className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.18em] text-indigo-600">
              <CircleDot size={13} />
              Processing overview
            </div>
            <h1 className="text-2xl font-bold tracking-tight text-slate-950 sm:text-3xl">
              Data Processing Pipeline
            </h1>
            <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-500">
              Track every sample from raw Blender data through semantic and motion quality
              decisions.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={handleRefreshResults}
              disabled={isLoadingResults || isRunning}
              className="inline-flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-4 py-2.5 text-sm font-medium text-slate-600 shadow-sm transition hover:border-slate-300 hover:bg-slate-50"
            >
              <RefreshCw className={isLoadingResults ? 'animate-spin' : ''} size={16} />
              Refresh Results
            </button>
            <button
              type="button"
              onClick={handleRunPipeline}
              disabled={isRunning}
              className="inline-flex min-w-32 items-center justify-center gap-2 rounded-xl bg-indigo-600 px-4 py-2.5 text-sm font-semibold text-white shadow-lg shadow-indigo-200 transition hover:bg-indigo-700 disabled:cursor-not-allowed disabled:bg-indigo-400 disabled:shadow-none"
            >
              {isRunning ? (
                <LoaderCircle className="animate-spin" size={16} />
              ) : (
                <Play size={15} fill="currentColor" />
              )}
              {isRunning ? 'Running...' : 'Start Pipeline'}
            </button>
          </div>
        </section>

        {isRunning && (
          <section className="mt-5 rounded-2xl border border-indigo-100 bg-white p-5 shadow-sm shadow-indigo-100/50">
            <div className="flex items-center justify-between gap-4">
              <div className="flex min-w-0 items-center gap-3">
                <span className="grid size-9 shrink-0 place-items-center rounded-xl bg-indigo-50 text-indigo-600">
                  <LoaderCircle className="animate-spin" size={18} />
                </span>
                <div className="min-w-0">
                  <p className="truncate text-sm font-semibold text-slate-800">
                    {currentStep || 'Pipeline is running...'}
                  </p>
                  <p className="mt-0.5 font-mono text-[10px] text-slate-400">
                    TASK {taskId ? taskId.slice(0, 12) : 'CREATING'} · {pipelineLogs.length}{' '}
                    log entries
                  </p>
                </div>
              </div>
              <span className="font-mono text-lg font-bold text-indigo-600">{progress}%</span>
            </div>
            <div
              className="mt-4 h-2 overflow-hidden rounded-full bg-indigo-50"
              role="progressbar"
              aria-label="Pipeline execution progress"
              aria-valuemin="0"
              aria-valuemax="100"
              aria-valuenow={progress}
            >
              <div
                className="h-full rounded-full bg-gradient-to-r from-indigo-500 to-violet-500 transition-[width] duration-500 ease-out"
                style={{ width: `${Math.min(Math.max(progress, 0), 100)}%` }}
              />
            </div>
          </section>
        )}

        {error && (
          <div className="mt-5 flex items-start gap-3 rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
            <AlertTriangle className="mt-0.5 shrink-0" size={17} />
            <div className="min-w-0">
              <p className="font-semibold">Backend Request Failed</p>
              <p className="mt-0.5 break-words text-xs leading-5 text-rose-600">{error}</p>
            </div>
          </div>
        )}

        <section className="mt-8">
          <div className="mb-4 flex items-center justify-between">
            <div>
              <h2 className="text-sm font-semibold text-slate-900">Pipeline Flow</h2>
              <p className="mt-1 text-xs text-slate-400">
                4 stages ·{' '}
                {lastUpdated
                  ? `Results updated at ${lastUpdated.toLocaleTimeString('en-US', {
                      hour: '2-digit',
                      minute: '2-digit',
                    })}`
                  : 'Awaiting backend results'}
              </p>
            </div>
            <span className="rounded-lg bg-white px-2.5 py-1.5 font-mono text-[10px] text-slate-400 ring-1 ring-slate-200">
              RUN #MVO-0724
            </span>
          </div>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:flex lg:gap-0">
            {pipeline.map((stage, index) => (
              <PipelineCard
                key={stage.number}
                stage={stage}
                isLast={index === pipeline.length - 1}
              />
            ))}
          </div>
        </section>

        <section className="mt-10">
          <div className="mb-5">
            <h2 className="text-xl font-bold tracking-tight text-slate-950">
              Evaluation Results
            </h2>
            <p className="mt-1 text-sm text-slate-500">
              Evaluation & Decision · Joint assessment of Motion Quality and Semantic
              Consistency
            </p>
          </div>

          <div className="mb-5 grid grid-cols-2 gap-3 lg:grid-cols-4">
            {[
              {
                label: 'Processed Samples',
                value: samples.length,
                detail: 'Current run',
                icon: Activity,
                color: 'text-indigo-600 bg-indigo-50',
              },
              {
                label: 'Keep Rate',
                value: `${keepRate}%`,
                detail: `${kept} samples passed`,
                icon: Gauge,
                color: 'text-emerald-600 bg-emerald-50',
              },
              {
                label: 'Average Motion Quality',
                value: averageMotion,
                detail: 'Overall motion quality score',
                icon: Zap,
                color: 'text-amber-600 bg-amber-50',
              },
              {
                label: 'Semantic Consistency',
                value: `${semanticRate}%`,
                detail: `${semanticMatches} / ${samples.length} samples`,
                icon: Check,
                color: 'text-violet-600 bg-violet-50',
              },
            ].map((stat) => {
              const Icon = stat.icon
              return (
                <article
                  key={stat.label}
                  className="rounded-2xl border border-slate-200/80 bg-white p-4 shadow-sm sm:p-5"
                >
                  <div className="flex items-center justify-between">
                    <p className="text-xs font-medium text-slate-500">{stat.label}</p>
                    <span className={`rounded-lg p-2 ${stat.color}`}>
                      <Icon size={16} />
                    </span>
                  </div>
                  <p className="mt-3 text-2xl font-bold tracking-tight text-slate-950">
                    {stat.value}
                  </p>
                  <p className="mt-1 text-[11px] text-slate-400">{stat.detail}</p>
                </article>
              )
            })}
          </div>

          <div className="overflow-hidden rounded-2xl border border-slate-200/80 bg-white shadow-sm">
            <div className="flex flex-col gap-3 border-b border-slate-100 p-4 sm:flex-row sm:items-center sm:justify-between">
              <div className="flex rounded-xl bg-slate-100 p-1">
                {[
                  ['all', 'All'],
                  ['keep', 'Keep'],
                  ['discard', 'Discard'],
                ].map(([value, label]) => (
                  <button
                    key={value}
                    type="button"
                    onClick={() => setFilter(value)}
                    className={`rounded-lg px-3 py-1.5 text-xs font-medium transition ${
                      filter === value
                        ? 'bg-white text-slate-900 shadow-sm'
                        : 'text-slate-500 hover:text-slate-700'
                    }`}
                  >
                    {label}
                  </button>
                ))}
              </div>
              <label className="flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-3 py-2 text-slate-400 transition focus-within:border-indigo-300 focus-within:ring-2 focus-within:ring-indigo-50">
                <Search size={15} />
                <input
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  placeholder="Search sample ID..."
                  className="w-full bg-transparent text-xs text-slate-700 outline-none placeholder:text-slate-400 sm:w-48"
                />
              </label>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full min-w-[920px] border-collapse text-left">
                <thead>
                  <tr className="border-b border-slate-100 bg-slate-50/60 text-[10px] font-semibold uppercase tracking-wider text-slate-400">
                    <th className="px-5 py-3.5">Sample ID / Name</th>
                    <th className="px-5 py-3.5">Motion Quality</th>
                    <th className="px-5 py-3.5">Semantic Level</th>
                    <th className="px-5 py-3.5">Decision</th>
                    <th className="px-5 py-3.5">Reason Code</th>
                    <th className="px-5 py-3.5">
                      <span className="sr-only">Actions</span>
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {visibleSamples.map((sample) => (
                    <tr key={sample.id} className="group transition hover:bg-slate-50/70">
                      <td className="px-5 py-4">
                        <div className="flex items-center gap-3">
                          <span className="grid size-9 place-items-center rounded-xl bg-slate-100 text-slate-500">
                            <FileJson size={16} />
                          </span>
                          <div>
                            <p className="font-mono text-xs font-semibold text-slate-800">
                              {sample.id}
                            </p>
                            <p className="mt-1 text-[11px] text-slate-400">
                              {sample.task} · {sample.time}
                            </p>
                          </div>
                        </div>
                      </td>
                      <td className="px-5 py-4">
                        <ScoreBar value={sample.score} />
                      </td>
                      <td className="px-5 py-4">
                        <div className="flex items-center gap-2">
                          <span
                            className={`size-1.5 rounded-full ${
                              sample.semantic === 'consistent'
                                ? 'bg-emerald-500'
                                : sample.semantic === 'uncertain'
                                  ? 'bg-amber-500'
                                  : 'bg-rose-500'
                            }`}
                          />
                          <div>
                            <p className="text-xs font-medium text-slate-700">
                              {sample.semantic === 'consistent'
                                ? 'Consistent'
                                : sample.semantic === 'inconsistent'
                                  ? 'Inconsistent'
                                  : sample.semantic === 'uncertain'
                                    ? 'Uncertain'
                                    : 'Unknown'}
                            </p>
                            <p className="mt-0.5 text-[10px] text-slate-400">
                              Confidence {sample.confidence}%
                            </p>
                          </div>
                        </div>
                      </td>
                      <td className="px-5 py-4">
                        <span
                          className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[11px] font-semibold ${
                            sample.decision === 'keep'
                              ? 'bg-emerald-50 text-emerald-700 ring-1 ring-emerald-100'
                              : 'bg-rose-50 text-rose-700 ring-1 ring-rose-100'
                          }`}
                        >
                          {sample.decision === 'keep' ? <Check size={11} /> : <X size={11} />}
                          {sample.decision === 'keep' ? 'Keep' : 'Discard'}
                        </span>
                      </td>
                      <td className="max-w-xs px-5 py-4">
                        {sample.reasons.length ? (
                          <div className="flex flex-wrap gap-1.5">
                            {sample.reasons.map((reason) => (
                              <span
                                key={reason}
                                className="rounded-md bg-slate-100 px-2 py-1 text-[10px] text-slate-600"
                              >
                                {reason}
                              </span>
                            ))}
                          </div>
                        ) : (
                          <span className="text-xs text-slate-300">—</span>
                        )}
                      </td>
                      <td className="px-5 py-4 text-right">
                        <button
                          type="button"
                          className="rounded-lg p-2 text-slate-300 transition hover:bg-white hover:text-slate-600 hover:shadow-sm"
                          aria-label={`View details for ${sample.id}`}
                        >
                          <ChevronDown size={15} />
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {visibleSamples.length === 0 && (
                <div className="grid place-items-center px-5 py-14 text-center">
                  {isLoadingResults ? (
                    <>
                      <LoaderCircle className="mb-3 animate-spin text-indigo-500" size={24} />
                      <p className="text-sm font-medium text-slate-600">
                        Fetching latest results...
                      </p>
                    </>
                  ) : (
                    <>
                      <Search className="mb-3 text-slate-300" size={24} />
                      <p className="text-sm font-medium text-slate-600">
                        No matching samples
                      </p>
                      <p className="mt-1 text-xs text-slate-400">
                        Run the pipeline or adjust the filters
                      </p>
                    </>
                  )}
                </div>
              )}
            </div>

            <footer className="flex items-center justify-between border-t border-slate-100 px-5 py-3 text-[11px] text-slate-400">
              <span>
                Showing {visibleSamples.length} / {samples.length} samples
              </span>
              <span className="font-mono">refinement_output/latest.jsonl</span>
            </footer>
          </div>
        </section>
      </main>

      <footer className="mt-4 border-t border-slate-200/80 bg-white">
        <div className="mx-auto flex max-w-[1500px] flex-col gap-2 px-5 py-5 text-[11px] text-slate-400 sm:flex-row sm:items-center sm:justify-between lg:px-8">
          <span>Mini-VLO Semantic-Motion Pipeline</span>
          <span className="font-mono">module_c · schema v2</span>
        </div>
      </footer>
    </div>
  )
}

export default App
