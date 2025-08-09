import { API_BASE } from './api'

export function openEventSource(path) {
    return new EventSource(API_BASE + path)
}
