// =============== 影片ID處理工具組合式函數 ===============
// 功能：提供影片URL的ID提取、標籤生成、階段狀態轉換等工具函數

export function useVideoId() {
    // ──────────────── 影片標籤生成函數 ────────────────
    // 根據影片URL生成人類可讀的標籤文字
    function labelFor(url) {
        if (!url) return '未知'
        try {
            const u = new URL(url)
            const host = u.hostname.replace(/^www\./, '')  // 移除 www 前綴

            // YouTube 影片處理
            if (host.includes('youtube.com') || host.includes('youtu.be')) {
                const vid = u.searchParams.get('v') || u.pathname.split('/').filter(Boolean).pop()
                return vid ? `YouTube: ${vid}` : `YouTube: ?`
            }

            // BiliBili 影片處理
            if (host.includes('bilibili.com') || host.includes('b23.tv')) {
                const m = u.pathname.match(/(BV[0-9A-Za-z]{10})/)  // 匹配BV號格式
                const id = (m && m[1]) || u.pathname.split('/').filter(Boolean).pop()
                return id ? `BiliBili: ${id}` : `BiliBili: ?`
            }

            // 其他網站的通用處理
            const tail = u.pathname.split('/').filter(Boolean).pop()
            return tail ? `${host}: ${tail}` : host
        } catch {
            // URL 解析失敗時的後備處理
            const s = String(url)
            return s.length > 36 ? s.slice(0, 33) + '…' : s
        }
    }

    // ──────────────── 影片ID提取函數 ────────────────
    // 從影片URL中提取唯一的識別ID，用於任務去重和關聯
    function videoId(u) {
        try {
            const url = new URL(u)
            const host = url.hostname.replace(/^www\./, '')

            // youtu.be 短連結處理
            if (host.includes('youtu.be')) return url.pathname.split('/').filter(Boolean).pop() || u

            // youtube.com 標準連結處理
            if (host.includes('youtube.com')) return url.searchParams.get('v') || url.pathname.split('/').filter(Boolean).pop() || u

            // BiliBili 影片處理
            if (host.includes('bilibili.com') || host.includes('b23.tv')) {
                const m = url.pathname.match(/(BV[0-9A-Za-z]{10})/)  // 匹配BV號格式
                return (m && m[1]) || u
            }

            // 其他網站的通用處理：使用 host + pathname 生成雜湊ID
            const base = (host + ':' + url.pathname)
            return simpleHash(base).slice(0, 12)
        } catch {
            // URL 解析失敗時，直接對原始字串進行雜湊
            return simpleHash(String(u)).slice(0, 12)
        }
    }

    // ──────────────── 簡單雜湊函數 ────────────────
    // 生成字串的簡單雜湊值，用於非標準影片網站的ID生成
    function simpleHash(str) {
        let h = 0
        for (let i = 0; i < str.length; i++) {
            h = ((h << 5) - h) + str.charCodeAt(i);
            h |= 0  // 轉換為32位整數
        }
        return Math.abs(h).toString(36)  // 轉換為36進制字串
    }

    // ──────────────── 階段狀態轉換函數 ────────────────
    // 將後端的技術階段名稱轉換為人類可讀的中文狀態描述
    function humanPhase(phase, pct) {
        // 階段名稱對照表
        const label = {
            queued: '待處理',      // 佇列中
            download: '下載中',    // 下載影片
            transcribe: '轉錄中',  // 語音轉文字
            subtitle: '字幕解析',  // 解析字幕檔案
            extract: '抽幀中',     // 提取影片幀
            audio: '音訊比對',     // 音訊相似度比對
            image: '畫面比對',     // 畫面相似度比對
            text: '文本比對',      // 文本相似度比對
            compare: '比對中'      // 綜合比對
        }[phase] || '處理中'

        // 確保百分比在 0-100 範圍內並四捨五入
        const n = Math.max(0, Math.min(100, Math.round(pct || 0)))
        return `${label} ${n}%`
    }

    // =============== 返回值 ===============
    return {
        labelFor,      // 影片標籤生成
        videoId,       // 影片ID提取
        humanPhase     // 階段狀態轉換
    }
}
