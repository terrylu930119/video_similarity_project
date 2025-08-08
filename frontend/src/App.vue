<template>
  <div class="container">
    <h1 class="title">影音比對系統</h1>
    <p class="subtitle">輸入參考影片與多支待比對影片，設定參數後開始分析；可即時查看進度與結果。</p>

    <div class="grid grid-2">
      <!-- 操作面板 -->
      <section class="card">
        <h3>輸入與參數</h3>

        <label>參考影片（Reference URL）</label>
        <input v-model.trim="refUrl" type="text" placeholder="https://youtu.be/...." />

        <label style="margin-top:14px">要比對的影片（多個）</label>
        <textarea v-model="listInput" placeholder="可多行貼上，逗號或空白自動分割。"></textarea>
        <div class="row" style="align-items:center;gap:10px">
          <button class="btn" @click="addChips">加入</button>
          <span class="hint">已加入 <strong>{{ chips.length }}</strong> 支影片（自動去重、可刪除）</span>
        </div>

        <div class="chips" aria-label="chips">
          <div v-for="(url, i) in chips" :key="url + i" class="chip">
            <span class="chip-text">{{ labelFor(url) }}</span>
            <span class="x" title="移除" @click="removeChip(i)">×</span>
          </div>
        </div>

        <div class="row" style="margin-top:14px">
          <div style="min-width:200px;flex:1">
            <label>幀間隔</label>
            <select v-model="interval">
              <option value="auto">自動</option>
              <option value="0.5">0.5 秒</option>
              <option value="1">1 秒</option>
              <option value="2">2 秒</option>
              <option value="5">5 秒</option>
            </select>
          </div>
          <div style="flex:1"></div>
        </div>

        <div class="row" style="justify-content:space-between;margin-top:10px">
          <label class="checkbox">
            <input v-model="keep" type="checkbox" />
            保留中間檔案（Debug 用）
          </label>
          <div class="btns">
            <button class="btn" @click="clearAll">清除</button>
            <button class="btn primary" :disabled="!canStart || loading" @click="submit">
              {{ loading ? '處理中…' : '開始比對' }}
            </button>
          </div>
        </div>

        <div v-if="tasks.length" style="margin-top:16px">
          <div class="row" style="justify-content:space-between;margin-bottom:6px">
            <div class="tiny">整體進度｜已完成 {{ doneCount }} / {{ totalCount }}</div>
            <div class="tiny">{{ overallPercent }}%</div>
          </div>
          <div class="progress"><div class="bar" :style="{ width: overallPercent + '%' }"></div></div>
        </div>
      </section>

      <!-- 任務佇列 -->
      <aside class="card">
        <div class="row" style="justify-content:space-between;align-items:center">
          <h3 style="margin:0">任務佇列</h3>
          <div class="sticky-actions">
            <button class="btn" @click="stopAll" :disabled="!tasks.length">停止全部</button>
          </div>
        </div>

        <div class="queue" style="display:grid;gap:10px;margin-top:10px">
          <div v-if="!tasks.length" class="hint">尚無任務</div>

          <div v-for="t in tasks" :key="t.id" class="queue-item">
            <div class="q-head">
              <div class="q-url" :title="t.url">{{ labelFor(t.url) }}</div>
              <div class="q-right tiny">{{ t.status }}</div>
            </div>
            <div class="progress tiny"><div class="bar" :style="{ width: (t.progress || 0) + '%' }"></div></div>
            <div class="q-actions">
              <button class="btn small" @click="t.showLog = !t.showLog">{{ t.showLog ? '隱藏 log' : '顯示 log' }}</button>
              <button class="btn small" @click="cancelTask(t)" :disabled="!running">取消</button>
            </div>
            <pre v-if="t.showLog" class="log" aria-live="polite">{{ t.log.join('\n') }}</pre>
          </div>
        </div>
      </aside>
    </div>

    <!-- 結果 -->
    <section class="card" style="margin-top:16px">
      <h3>比對結果</h3>
      <div class="results-grid" style="margin-top:12px">
        <div v-if="!results.length" class="hint">尚未有完成結果</div>

        <div v-for="r in sortedResults" :key="r.url" class="result">
          <div class="meta">
            <div>
              <div class="tiny">{{ labelFor(r.url) }}</div>
              <div class="score">{{ r.score }}%</div>
            </div>
          </div>
          <div class="divider"></div>
          <div class="tiny">音訊 {{ r.audio }}｜畫面 {{ r.visual }}｜內容 {{ r.text }}</div>
        </div>
      </div>
    </section>
  </div>
</template>

<script setup>
import { reactive, ref, computed, onBeforeUnmount, nextTick } from 'vue'
import axios from 'axios'

// ====== 表單狀態 ======
const refUrl = ref('')
const listInput = ref('')
const chips = ref([])
const interval = ref('auto')
const keep = ref(false)

// ====== 執行狀態 ======
const loading = ref(false)
const running = ref(false)
const tasks = reactive([])
const results = reactive([])
let es = null

// ====== 常數 ======
const API_BASE = 'http://localhost:8000'
const COMPARE_PATH = '/api/compare'
const EVENTS_PATH = '/api/events'
const STATUS_PATH = '/api/status'

// ====== 派生 ======
const totalCount = computed(() => tasks.length) // 含參考 + 所有比對影片
const doneCount  = computed(() => tasks.filter(t => t.status === '完成').length)
const overallPercent = computed(() => {
  if (!tasks.length) return 0
  const sum = tasks.reduce((acc, t) => acc + Math.min(t.progress || 0, 100), 0)
  return Math.round(sum / tasks.length)
})

const canStart = computed(() => !!refUrl.value && chips.value.length > 0)
const sortedResults = computed(() => [...results].sort((a, b) => (b.score ?? 0) - (a.score ?? 0)))

// ====== 工具 ======
function normalizeListInput () {
  const raw = listInput.value
    .split(/[\n,\s]+/g)
    .map(s => s.trim())
    .filter(Boolean)
  const set = new Set([...(chips.value || []), ...raw])
  chips.value = [...set]
  listInput.value = ''
}
function addChips () { normalizeListInput() }
function removeChip (i) { chips.value.splice(i, 1) }

function labelFor (url) {
  try {
    const u = new URL(url);
    const host = u.hostname.replace(/^www\./, '');
    if (host.includes('youtube.com') || host.includes('youtu.be')) {
      const vid = u.searchParams.get('v') || u.pathname.split('/').filter(Boolean).pop();
      return vid ? `YouTube: ${vid}` : `YouTube: ?`;
    }
    if (host.includes('bilibili.com') || host.includes('b23.tv')) {
      const m = u.pathname.match(/(BV[a-zA-Z0-9]{10})/);
      const id = (m && m[1]) || u.pathname.split('/').filter(Boolean).pop();
      return id ? `Bili: ${id}` : `Bili: ?`;
    }
    const tail = u.pathname.split('/').filter(Boolean).pop();
    return tail ? `${host}: ${tail}` : host;
  } catch {
    return url.length > 36 ? url.slice(0, 33) + '…' : url;
  }
}

function videoId(u) {
  try {
    const url = new URL(u)
    const host = url.hostname.replace(/^www\./, '')
    if (host.includes('youtu.be')) return url.pathname.split('/').filter(Boolean).pop() || u
    if (host.includes('youtube.com')) return url.searchParams.get('v') || url.pathname.split('/').filter(Boolean).pop() || u
    if (host.includes('bilibili.com') || host.includes('b23.tv')) {
      const m = url.pathname.match(/(BV[a-zA-Z0-9]+)/)
      return (m && m[1]) || u
    }
    return (host + ':' + url.pathname).slice(0, 64)
  } catch {
    return u.slice(0, 64)
  }
}

function pushLog (t, line) {
  const now = new Date().toLocaleTimeString()
  t.log.push(`[${now}] ${line}`)
  if (t.log.length > 400) t.log.splice(0, t.log.length - 400)
  nextTick(() => {
    const box = document.querySelector(`pre.log:last-of-type`)
    if (box) box.scrollTop = box.scrollHeight
  })
}

function clearAll () {
  refUrl.value = ''
  listInput.value = ''
  chips.value = []
  tasks.splice(0, tasks.length)
  results.splice(0, results.length)
}

async function stopAll () {
  if (!tasks.length) return
  try {
    const ids = tasks.map(t => t.id).filter(Boolean)
    await axios.post(`${API_BASE}/api/cancel`, { task_ids: ids })
  } catch {}
}

async function cancelTask(t) {
  try {
    await axios.post(`${API_BASE}/api/cancel`, { task_ids: [t.id] })
  } catch {}
}

// ====== 狀態預讀（只影響 UI） ======
async function preloadStatus() {
  if (!refUrl.value) return
  try {
    const { data } = await axios.post(`${API_BASE}${STATUS_PATH}`, {
      ref: refUrl.value.trim(),
      comp: chips.value
    })
    const ensureTask = (url) => {
      const vid = videoId(url)
      let t = tasks.find(x => videoId(x.url) === vid)
      if (!t) {
        t = { id: 'pre-' + vid, url, ref: refUrl.value, isRef: false, progress: 0, status: '待處理', log: [], showLog: false }
        tasks.push(t)
      }
      return t
    }
    for (const s of data) {
      const t = ensureTask(s.url)
      const pct = Number(s.percent || 0)
      t.progress = Math.max(Number(t.progress || 0), pct)
      t.status = humanPhase(s.phase, pct)
    }
  } catch (e) {
    console.error('status preload failed', e)
  }
}

function humanPhase(phase, pct){
  const label = {
    queued:'待處理', download:'下載中', transcribe:'轉錄中',
    extract:'抽幀中', audio:'音訊比對', image:'畫面比對', text:'文本比對', compare:'比對中'
  }[phase] || '處理中'
  const n = Math.max(0, Math.min(100, Math.round(pct || 0)))
  return `${label} ${n}%`
}

// ====== SSE ======
function findTaskByEvent(e) {
  if (e.task_id) {
    const t = tasks.find(x => x.id === e.task_id)
    if (t) return t
  }
  if (e.url) {
    const vid = videoId(e.url)
    const t = tasks.find(x => videoId(x.url) === vid)
    if (t) return t
  }
  return null
}

function connectEvents () {
  if (es) { try { es.close() } catch {} es = null }
  es = new EventSource(`${API_BASE}${EVENTS_PATH}`)
  es.onmessage = (evt) => {
    if (!evt?.data) return
    let e = {}
    try { e = JSON.parse(evt.data) } catch { return }

    if (e.type === 'progress') {
      const t = findTaskByEvent(e)
      if (t) {
        const p = Number.isFinite(+e.percent) ? +e.percent : (t.progress || 0)
        t.progress = Math.max(Number(t.progress || 0), p)
        t.status = humanPhase(e.phase, t.progress)
        if (e.msg) pushLog(t, e.msg)
        if (p >= 100 || e.phase === 'compare' && p >= 100) t.status = '完成'
      }
    } else if (e.type === 'log') {
      const t = findTaskByEvent(e)
      if (t && e.msg) pushLog(t, e.msg)

    } else if (e.type === 'done') {
      const t = findTaskByEvent(e)
      if (t) { t.status = '完成'; t.progress = 100 }
      if (e.url) {
        const idx = results.findIndex(r => videoId(r.url) === videoId(e.url))
        const row = {
          pair: `${(t?.ref || '參考')} vs ${(t?.url || e.url || '未知')}`,
          url: e.url,
          ref: t?.ref || e.ref_url || refUrl.value,
          score: Math.round(Number(e.score || 0)),
          visual: Number(e.visual || 0).toFixed(2),
          audio:  Number(e.audio  || 0).toFixed(2),
          text:   Number(e.text   || 0).toFixed(2),
          hot: Array.isArray(e.hot) ? e.hot.join('、') : ''
        }
        if (idx >= 0) results.splice(idx, 1, row)
        else results.push(row)
      }
      if (tasks.every(x => x.status === '完成' || x.status === '失敗' || x.status === '已取消')) running.value = false
      finalizeRefCardIfAllDone()

    } else if (e.type === 'canceled') {
      const t = findTaskByEvent(e)
      if (t) { t.status = '已取消'; t.progress = 0; pushLog(t, '已取消') }
      finalizeRefCardIfAllDone()
    }
  }
  es.onerror = () => { es && es.close(); es = null }
}

function finalizeRefCardIfAllDone () {
  const normal = tasks.filter(t => !t.isRef)
  if (!normal.length) return
  if (normal.every(t => t.status === '完成' || t.status === '失敗' || t.status === '已取消')) {
    const refTask = tasks.find(t => t.isRef)
    if (refTask) { refTask.status = '完成'; refTask.progress = 100 }
  }
}

onBeforeUnmount(() => { if (es) try { es.close() } catch {}; es = null })

// ====== 送單 ======
async function submit () {
  if (!canStart.value) return
  loading.value = true
  results.splice(0, results.length)
  tasks.splice(0, tasks.length)

  const refTaskId = 'ref-' + videoId(refUrl.value)
  tasks.push({ id: refTaskId, url: refUrl.value, ref: refUrl.value, isRef: true, progress: 0, status: '佇列中', log: [], showLog: false })

  await preloadStatus()

  try {
    const res = await axios.post(`${API_BASE}${COMPARE_PATH}`, {
      ref: refUrl.value.trim(),
      comp: chips.value,
      interval: interval.value,
      keep: keep.value
    })
    const data = res.data || {}
    if (Array.isArray(data.task_ids)) {
      data.task_ids.forEach(({ url, task_id, ref_url }) => {
        const vid = videoId(url)
        const exist = tasks.find(x => !x.isRef && videoId(x.url) === vid)
        if (exist) { exist.id = task_id; exist.ref = ref_url; if (!exist.status) exist.status = '佇列中' }
        else tasks.push({ id: task_id, url, ref: ref_url, isRef: false, progress: 0, status: '佇列中', log: [], showLog: false })
      })
      running.value = true
      connectEvents()
    }
  } catch (e) {
    console.error(e)
  } finally {
    loading.value = false
    setTimeout(() => window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' }), 300)
  }
}
</script>

<style>
:root{
  --panel:#101317;--panel-2:#0b0e12;--border:#20242c;--muted:#a2a9b6;--primary:#5aa4ff;--ok:#4cc38a;
}
*{box-sizing:border-box}
html,body{margin:0;padding:0;background:#0a0d11;color:#e7ecf4;
  font:16px/1.5 ui-sans-serif,system-ui,-apple-system,"PingFang TC","Noto Sans TC",Segoe UI,Roboto,Helvetica,Arial;
  overflow-x: hidden;
}
.container{max-width:1100px;margin:40px auto;padding:0 20px;}
.title{font-size:28px;font-weight:800;letter-spacing:.2px;margin:0 0 6px}
.subtitle{color:var(--muted);margin:0 0 20px}

.grid{display:grid;gap:16px}
@media (min-width: 960px){.grid-2{grid-template-columns: 1.2fr .8fr}}

.card{background:var(--panel);border:1px solid var(--border);
  border-radius:14px;padding:18px;box-shadow:0 10px 24px rgba(0,0,0,.2)}
.card h3{margin:0 0 12px;font-size:18px}
label{display:block;margin:10px 0 6px;font-weight:600}
input[type=text], textarea, select{
  width:100%;background:var(--panel-2);border:1px solid var(--border);
  color:#e8eef7;border-radius:10px;padding:10px 12px;outline:none}
textarea{min-height:90px;resize:vertical}
.row{display:flex;gap:10px;align-items:center}
.btns{display:flex;gap:8px}
.btn{background:#171b22;border:1px solid var(--border);padding:8px 12px;border-radius:10px;color:#dfe6f2;cursor:pointer}
.btn:hover{background:#1a2029}
.btn.primary{background:linear-gradient(180deg,#24364a,#1a2838);border-color:#2f3a4a;color:#eaf2ff}
.btn.small{font-size:12px;padding:6px 10px}
.checkbox{display:flex;gap:8px;align-items:center}
.checkbox.small{font-size:12px}
.hint{color:var(--muted);font-size:13px}

.chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:8px}
.chip{display:flex;gap:6px;align-items:center;background:#0e1217;border:1px solid var(--border);border-radius:999px;padding:6px 10px}
.chip-text{font-size:12px;color:#d9e2ef}
.chip .x{cursor:pointer;opacity:.7}
.chip .x:hover{opacity:1}

.queue-item{background:#0c1015;border:1px solid var(--border);border-radius:12px;padding:10px}
.q-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}
.q-url{font-weight:600;max-width:70%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.q-right{color:var(--muted)}
.progress{height:8px;background:#0c0f14;border:1px solid #1c222c;border-radius:999px;overflow:hidden}
.progress .bar{height:100%;background:linear-gradient(90deg,#3c83f6,#5aa4ff)}
.q-actions{display:flex;justify-content:space-between;align-items:center;margin-top:8px}

.results-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}
.result{background:#0c1015;border:1px solid var(--border);border-radius:12px;padding:10px}
.result .meta{display:flex;justify-content:space-between;align-items:center}
.result .score{font-size:28px;font-weight:800;color:#eaf2ff}
.divider{height:1px;background:linear-gradient(90deg,transparent,#30394a,transparent);margin:8px 0}
.tiny{color:#aeb7c6;font-size:12px}

/* Log 區：斷行＋美化捲軸，避免撐爆畫面 */
pre.log{
  max-height:220px;overflow:auto;background:#0a0e13;border:1px solid var(--border);
  padding:10px;border-radius:10px;font:12px/1.4 ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,"Liberation Mono","Courier New",monospace;color:#cfd7e6;
  white-space:pre-wrap;word-break:break-word;
}
pre.log::-webkit-scrollbar{ height:8px; width:8px; }
pre.log::-webkit-scrollbar-track{ background:#0d1218; border-radius:10px; }
pre.log::-webkit-scrollbar-thumb{ background:#2a3342; border-radius:10px; }
pre.log::-webkit-scrollbar-thumb:hover{ background:#364154; }
</style>
