import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react'
import { fetchSupabaseTable, isSupabaseConfigured } from '@/lib/supabase'

export interface AppEventRecord extends Record<string, any> {
  event_id?: string
  created_at?: string
}

export interface ChatMessageRecord {
  message_id: string
  created_at: string
  session_id: string | null
  device_id: string | null
  role: string
  content: string | null
}

type DataErrors = {
  events: string | null
  chatMessages: string | null
}

type AppDataContextValue = {
  events: AppEventRecord[]
  chatMessages: ChatMessageRecord[]
  loading: boolean
  errors: DataErrors
  supabaseAvailable: boolean
  refreshEvents: () => Promise<void>
  refreshChatMessages: () => Promise<void>
  refreshAll: () => Promise<void>
}

const AppDataContext = createContext<AppDataContextValue | undefined>(undefined)

const eventsTableName = import.meta.env.VITE_SUPABASE_EVENTS_TABLE?.trim() || 'events'
const eventsOrderColumn = import.meta.env.VITE_SUPABASE_EVENTS_ORDER_COLUMN?.trim() || 'created_at'

const chatMessagesTableName = import.meta.env.VITE_SUPABASE_CHAT_MESSAGES_TABLE?.trim() || 'chat_messages'
const chatMessagesOrderColumn = import.meta.env.VITE_SUPABASE_CHAT_MESSAGES_ORDER_COLUMN?.trim() || 'created_at'

const notConfiguredMessage =
  'Supabase configuration is missing. Please set VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY to enable remote data loading.'

export const AppDataProvider = ({ children }: { children: React.ReactNode }) => {
  const [events, setEvents] = useState<AppEventRecord[]>([])
  const [chatMessages, setChatMessages] = useState<ChatMessageRecord[]>([])
  const [loading, setLoading] = useState(false)
  const [errors, setErrors] = useState<DataErrors>({ events: null, chatMessages: null })

  const supabaseAvailable = isSupabaseConfigured()
  const isMountedRef = useRef(true)

  useEffect(() => {
    isMountedRef.current = true
    return () => {
      isMountedRef.current = false
    }
  }, [])

  const updateError = useCallback((key: keyof DataErrors, value: string | null) => {
    if (!isMountedRef.current) return
    setErrors((prev) => ({ ...prev, [key]: value }))
  }, [])

  const loadEvents = useCallback(async () => {
    if (!supabaseAvailable) {
      updateError('events', notConfiguredMessage)
      if (isMountedRef.current) {
        setEvents([])
      }
      return
    }

    try {
      const data = await fetchSupabaseTable<AppEventRecord>(eventsTableName, {
        orderBy: eventsOrderColumn,
        ascending: false
      })

      if (!isMountedRef.current) return

      setEvents(data)
      updateError('events', null)
    } catch (error) {
      console.error('Failed to load events from Supabase', error)
      updateError('events', error instanceof Error ? error.message : 'Failed to load events.')
    }
  }, [supabaseAvailable, updateError, eventsOrderColumn, eventsTableName])

  const loadChatMessages = useCallback(async () => {
    if (!supabaseAvailable) {
      updateError('chatMessages', notConfiguredMessage)
      if (isMountedRef.current) {
        setChatMessages([])
      }
      return
    }

    try {
      const data = await fetchSupabaseTable<ChatMessageRecord>(chatMessagesTableName, {
        orderBy: chatMessagesOrderColumn,
        ascending: false
      })

      if (!isMountedRef.current) return

      setChatMessages(data)
      updateError('chatMessages', null)
    } catch (error) {
      console.error('Failed to load chat messages from Supabase', error)
      updateError('chatMessages', error instanceof Error ? error.message : 'Failed to load chat messages.')
    }
  }, [supabaseAvailable, updateError, chatMessagesOrderColumn, chatMessagesTableName])

  useEffect(() => {
    if (!supabaseAvailable) {
      setLoading(false)
      updateError('events', notConfiguredMessage)
      updateError('chatMessages', notConfiguredMessage)
      return
    }

    let cancelled = false

    const loadData = async () => {
      setLoading(true)
      await Promise.allSettled([loadEvents(), loadChatMessages()])
      if (!cancelled && isMountedRef.current) {
        setLoading(false)
      }
    }

    loadData()

    return () => {
      cancelled = true
    }
  }, [supabaseAvailable, loadEvents, loadChatMessages, updateError])

  const refreshEvents = useCallback(async () => {
    await loadEvents()
  }, [loadEvents])

  const refreshChatMessages = useCallback(async () => {
    await loadChatMessages()
  }, [loadChatMessages])

  const refreshAll = useCallback(async () => {
    await Promise.allSettled([loadEvents(), loadChatMessages()])
  }, [loadEvents, loadChatMessages])

  const value = useMemo<AppDataContextValue>(
    () => ({
      events,
      chatMessages,
      loading,
      errors,
      supabaseAvailable,
      refreshEvents,
      refreshChatMessages,
      refreshAll
    }),
    [events, chatMessages, loading, errors, supabaseAvailable, refreshEvents, refreshChatMessages, refreshAll]
  )

  return <AppDataContext.Provider value={value}>{children}</AppDataContext.Provider>
}

export const useAppData = () => {
  const context = useContext(AppDataContext)
  if (!context) {
    throw new Error('useAppData must be used within an AppDataProvider')
  }
  return context
}
