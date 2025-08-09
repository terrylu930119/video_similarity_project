import axios from 'axios'
import.meta.env.VITE_API_BASE

export const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

const http = axios.create({
    baseURL: API_BASE,
    headers: { 'Content-Type': 'application/json' }
})

export const compare = (payload) =>
    http.post('/api/compare', payload).then(r => r.data)

export const status = (payload) =>
    http.post('/api/status', payload).then(r => r.data)

export const cancel = (payload) =>
    http.post('/api/cancel', payload).then(r => r.data)
