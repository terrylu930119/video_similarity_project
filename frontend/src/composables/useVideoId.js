export function useVideoId() {
    function labelFor(url) {
        if (!url) return '未知'
        try {
            const u = new URL(url)
            const host = u.hostname.replace(/^www\./, '')
            if (host.includes('youtube.com') || host.includes('youtu.be')) {
                const vid = u.searchParams.get('v') || u.pathname.split('/').filter(Boolean).pop()
                return vid ? `YouTube: ${vid}` : `YouTube: ?`
            }
            if (host.includes('bilibili.com') || host.includes('b23.tv')) {
                const m = u.pathname.match(/(BV[0-9A-Za-z]{10})/)
                const id = (m && m[1]) || u.pathname.split('/').filter(Boolean).pop()
                return id ? `BiliBili: ${id}` : `BiliBili: ?`
            }
            const tail = u.pathname.split('/').filter(Boolean).pop()
            return tail ? `${host}: ${tail}` : host
        } catch {
            const s = String(url)
            return s.length > 36 ? s.slice(0, 33) + '…' : s
        }
    }

    function videoId(u) {
        try {
            const url = new URL(u)
            const host = url.hostname.replace(/^www\./, '')
            if (host.includes('youtu.be')) return url.pathname.split('/').filter(Boolean).pop() || u
            if (host.includes('youtube.com')) return url.searchParams.get('v') || url.pathname.split('/').filter(Boolean).pop() || u
            if (host.includes('bilibili.com') || host.includes('b23.tv')) {
                const m = url.pathname.match(/(BV[0-9A-Za-z]{10})/)
                return (m && m[1]) || u
            }
            const base = (host + ':' + url.pathname)
            return simpleHash(base).slice(0, 12)
        } catch {
            return simpleHash(String(u)).slice(0, 12)
        }
    }

    function simpleHash(str) {
        let h = 0
        for (let i = 0; i < str.length; i++) { h = ((h << 5) - h) + str.charCodeAt(i); h |= 0 }
        return Math.abs(h).toString(36)
    }

    function humanPhase(phase, pct) {
        const label = {
            queued: '待處理', download: '下載中',
            transcribe: '轉錄中', subtitle: '字幕解析',
            extract: '抽幀中', audio: '音訊比對',
            image: '畫面比對', text: '文本比對', compare: '比對中'
        }[phase] || '處理中'
        const n = Math.max(0, Math.min(100, Math.round(pct || 0)))
        return `${label} ${n}%`
    }

    return { labelFor, videoId, humanPhase }
}
