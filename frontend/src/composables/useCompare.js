import { reactive, ref, computed, onBeforeUnmount, nextTick } from 'vue'
import * as api from '@/services/api'
import { openEventSource } from '@/services/sse'
import { useVideoId } from './useVideoId'

export function useCompare() {
    // ====== 表單 ======
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

    const { videoId, labelFor, humanPhase } = useVideoId()

    // ====== 派生 ======
    const totalCount = computed(() => tasks.length)
    const doneCount = computed(() => tasks.filter(t => t.status === '完成').length)
    const overallPercent = computed(() => {
        if (!tasks.length) return 0
        const sum = tasks.reduce((acc, t) => acc + Math.min(t.progress || 0, 100), 0)
        return Math.round(sum / tasks.length)
    })
    const canStart = computed(() => {
        // 先看 reactive 的 refUrl
        let hasRef = !!refUrl.value;

        // 後備：若這一拍 reactive 還沒 flush，從輸入框 DOM 取一次
        if (!hasRef && typeof window !== 'undefined') {
            const refInput = document.querySelector('input[type="text"]');
            if (refInput && refInput.value && refInput.value.trim() !== '') {
                hasRef = true;
            }
        }

        return hasRef && chips.value.length > 0 && !loading.value;
    });

    const sortedResults = computed(() => [...results].sort((a, b) => (b.score ?? 0) - (a.score ?? 0)))

    // ====== 工具 ======
    function normalizeListInput(overrideText) {
        // 來源優先：參數 > state > DOM 後備
        let source = ''
        if (typeof overrideText === 'string' && overrideText !== '') {
            source = overrideText
        } else if (overrideText && typeof overrideText === 'object' && 'value' in overrideText && String(overrideText.value ?? '') !== '') {
            source = String(overrideText.value ?? '')
        } else if (typeof listInput.value === 'string' && listInput.value !== '') {
            source = listInput.value
        } else if (typeof window !== 'undefined') {
            const el = document.querySelector('textarea')
            if (el && el.value) source = el.value
        }

        const raw = String(source || '')
            .split(/[\n,\s]+/g)
            .map(s => s.trim())
            .filter(Boolean)

        if (!raw.length) {
            listInput.value = ''
            return
        }

        // 只針對「比對清單」去重（不會把與 refUrl 相同的濾掉）
        const merged = [...(chips.value || []), ...raw]
        const seen = new Set()
        const uniq = []

        for (const url of merged) {
            const id = (() => {
                try { return videoId(url) || url.trim() } catch { return url.trim() }
            })()
            if (seen.has(id)) continue
            seen.add(id)
            uniq.push(url)
        }

        chips.value = uniq
        listInput.value = ''
        console.log('[useCompare] normalizeListInput: chips =', chips.value)
    }


    const addChips = async (text) => {
        // 先記快照；如果 state 還沒同步，快照可能是空
        let refSnapshot = refUrl.value;

        // 如果子元件有帶 payload，暫存（沒有也沒關係）
        if (typeof text === 'string' && text !== '') {
            listInput.value = text;
        } else if (text && typeof text === 'object' && 'value' in text && String(text.value ?? '') !== '') {
            listInput.value = String(text.value ?? '');
        }

        // 若快照是空，再從 DOM 抄一份候補
        if (!refSnapshot && typeof window !== 'undefined') {
            const refInput = document.querySelector('input[type="text"]');
            if (refInput && refInput.value && refInput.value.trim() !== '') {
                refSnapshot = refInput.value.trim();
            }
        }

        // 等 v-model flush 完成
        await nextTick();

        // 這一拍如果 refUrl 被清掉，就用快照或 DOM 值救回
        if ((!refUrl.value || refUrl.value.trim() === '') && refSnapshot) {
            console.warn('[useCompare] refUrl cleared unexpectedly. Restoring snapshot.');
            refUrl.value = refSnapshot;
        } else if ((!refUrl.value || refUrl.value.trim() === '') && typeof window !== 'undefined') {
            const refInput = document.querySelector('input[type="text"]');
            const domVal = refInput?.value?.trim();
            if (domVal) refUrl.value = domVal;
        }

        console.log('[useCompare] addchips(after restore): listInput=', listInput.value, ' refUrl=', refUrl.value);

        // 解析 listInput → 寫入 chips → 清空 textarea
        normalizeListInput();
    };


    const removeChip = (i) => { console.log('[useCompare] removeChip', i); chips.value.splice(i, 1) }

    function pushLog(t, line) {
        const now = new Date().toLocaleTimeString()
        t.log.push(`[${now}] ${line}`)
        if (t.log.length > 400) t.log.splice(0, t.log.length - 400)
        nextTick(() => { t.__logScroll && t.__logScroll() })
    }

    function clearAll() {
        console.log('[useCompare] clearAll')
        refUrl.value = ''
        listInput.value = ''
        chips.value = []
        tasks.splice(0, tasks.length)
        results.splice(0, results.length)
    }

    async function stopAll() {
        console.log('[useCompare] stopAll called')
        if (!tasks.length) return
        try {
            const ids = tasks.map(t => t.id).filter(Boolean)
            if (ids.length) await api.cancel({ task_ids: ids })
        } catch (e) { console.log('[useCompare] stopAll error', e) }
    }

    async function cancelTask(t) {
        console.log('[useCompare] cancelTask', t && t.id)
        try { if (t?.id) await api.cancel({ task_ids: [t.id] }) } catch (e) { console.log('[useCompare] cancelTask error', e) }
    }

    // ====== 狀態預讀（只影響 UI） ======
    async function preloadStatus() {
        if (!refUrl.value) return
        try {
            const data = await api.status({ ref: refUrl.value.trim(), comp: chips.value })
            const ensureTask = (url) => {
                const vid = videoId(url)
                let t = tasks.find(x => videoId(x.url) === vid)
                if (!t) {
                    t = { id: 'pre-' + vid, url, ref: refUrl.value, isRef: false, progress: 0, status: '待處理', log: [], showLog: false }
                    t.display = labelFor(url)
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

    function finalizeRefCardIfAllDone() {
        const normal = tasks.filter(t => !t.isRef)
        if (!normal.length) return
        if (normal.every(t => ['完成', '失敗', '已取消'].includes(t.status))) {
            const refTask = tasks.find(t => t.isRef)
            if (refTask) { refTask.status = '完成'; refTask.progress = 100 }
        }
    }

    function connectEvents() {
        console.log('[useCompare] connectEvents')
        if (es) { try { es.close() } catch { } es = null }
        es = openEventSource('/api/events')
        es.onmessage = (evt) => {
            if (!evt?.data) return
            let e = {}
            try { e = JSON.parse(evt.data) } catch { return }

            if (e.type === 'progress') {
                const t = findTaskByEvent(e)
                if (t) {
                    const hint = (typeof e.overallHint === 'number') ? Math.round(e.overallHint * 100) : null
                    const p = Number.isFinite(+e.percent) ? +e.percent : (t.progress || 0)
                    const np = (hint ?? p)

                    // 平滑進度更新：避免進度倒退，除非是階段變更
                    const currentProgress = t.progress || 0
                    if (np >= currentProgress || e.phase !== t.currentPhase) {
                        t.progress = np
                        t.currentPhase = e.phase
                    }

                    // 狀態優先吃後端 phaseName（例如「字幕解析」「音訊處理」）
                    if (e.phaseName) t.status = `${e.phaseName} ${t.progress}%`
                    else t.status = humanPhase(e.phase, t.progress)
                    if (e.msg) pushLog(t, e.msg)
                    if (e.textSource) t.text_source = e.textSource   // 'subtitle' | 'asr'
                    if (e.text_skipped) { t.text_skipped = true; if (!t.text_status && e.text_status) t.text_status = e.text_status }
                    if (np >= 100 || (e.phase === 'compare' && np >= 100)) t.status = '完成'
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
                        audio: Number(e.audio || 0).toFixed(2),
                        text: Number(e.text || 0).toFixed(2),
                        text_meaningful: (typeof e.text_meaningful === 'boolean' ? e.text_meaningful : true),
                        text_status: (typeof e.text_status === 'string' ? e.text_status : ''),
                        hot: Array.isArray(e.hot) ? e.hot.join('、') : ''
                    }
                    if (idx >= 0) results.splice(idx, 1, row)
                    else results.push(row)
                }
                // 任務卡也存一份文本狀態（之後 QueuePanel 用得到）
                if (t) {
                    if (typeof e.text_meaningful === 'boolean') t.text_meaningful = e.text_meaningful
                    if (typeof e.text_status === 'string') t.text_status = e.text_status
                    if (t.text_meaningful === false) t.text_skipped = true
                }
                if (tasks.every(x => ['完成', '失敗', '已取消'].includes(x.status))) running.value = false
                finalizeRefCardIfAllDone()

            } else if (e.type === 'canceled') {
                const t = findTaskByEvent(e)
                if (t) { t.status = '已取消'; t.progress = 0; pushLog(t, '已取消') }
                finalizeRefCardIfAllDone()
            }
        }
        es.onerror = () => { console.log('[useCompare] SSE error/closed'); es && es.close(); es = null }
    }

    onBeforeUnmount(() => { if (es) try { es.close() } catch { }; es = null })

    // ====== 送單 ======
    async function submit() {
        // ⬅️ 先等 v-model (update:keep / update:interval) 的變更 flush 完成
        await nextTick()

        console.log('[useCompare] submit called', { ref: refUrl.value, chips: chips.value })

        if (!canStart.value) { console.log('[useCompare] submit ignored: canStart=false'); return }
        loading.value = true
        results.splice(0, results.length)
        tasks.splice(0, tasks.length)

        const refTaskId = 'ref-' + videoId(refUrl.value)
        const refDisplay = labelFor(refUrl.value)
        tasks.push({
            id: refTaskId,
            url: refUrl.value,
            ref: refUrl.value,
            isRef: true,
            progress: 0,
            status: '佇列中',
            currentPhase: 'queued',
            log: [],
            showLog: false,
            display: refDisplay
        })

        await preloadStatus()

        try {
            const payload = {
                ref: refUrl.value.trim(),
                comp: [...chips.value],
                interval: interval.value,
                keep: !!keep.value
            }
            console.log('[useCompare] compare payload =', payload)

            const data = await api.compare(payload)
            if (Array.isArray(data.task_ids)) {
                data.task_ids.forEach(({ url, task_id, ref_url }) => {
                    const vid = videoId(url)
                    const exist = tasks.find(x => !x.isRef && videoId(x.url) === vid)
                    const display = labelFor(url)
                    if (exist) {
                        exist.id = task_id;
                        exist.ref = ref_url;
                        exist.display = display;
                        if (!exist.status) exist.status = '佇列中'
                        if (!exist.currentPhase) exist.currentPhase = 'queued'
                    }
                    else tasks.push({
                        id: task_id,
                        url,
                        ref: ref_url,
                        isRef: false,
                        progress: 0,
                        status: '佇列中',
                        currentPhase: 'queued',
                        log: [],
                        showLog: false,
                        display
                    })
                })
                running.value = true
                connectEvents()
            }
        } catch (e) {
            console.error('[useCompare] submit error', e)
        } finally {
            loading.value = false
            setTimeout(() => window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' }), 300)
        }
    }

    return {
        // state
        refUrl, listInput, chips, interval, keep, loading, running, tasks, results,
        // derived
        canStart, sortedResults, overallPercent, doneCount, totalCount,
        // methods
        addChips, removeChip, clearAll, submit, stopAll, cancelTask,
        // utils
        labelFor
    }
}
