const SUPABASE_URL = import.meta.env.VITE_SUPABASE_URL?.trim()
const SUPABASE_ANON_KEY = import.meta.env.VITE_SUPABASE_ANON_KEY?.trim()

export type SupabaseConfig = {
  url: string
  apiKey: string
}

let cachedConfig: SupabaseConfig | null | undefined

export const getSupabaseConfig = (): SupabaseConfig | null => {
  if (cachedConfig !== undefined) {
    return cachedConfig
  }

  if (!SUPABASE_URL || !SUPABASE_ANON_KEY) {
    cachedConfig = null
    if (import.meta.env.DEV) {
      console.warn(
        '[AppDataProvider] Supabase environment variables are missing. Set VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY to enable remote data loading.'
      )
    }
    return cachedConfig
  }

  const normalizedUrl = SUPABASE_URL.replace(/\/$/, '')

  cachedConfig = {
    url: normalizedUrl,
    apiKey: SUPABASE_ANON_KEY
  }

  return cachedConfig
}

export const isSupabaseConfigured = () => getSupabaseConfig() !== null

export type FetchSupabaseTableOptions = {
  orderBy?: string
  ascending?: boolean
  limit?: number
  signal?: AbortSignal
}

/**
 * Fetches rows from a Supabase table using the REST interface.
 */
export const fetchSupabaseTable = async <T extends Record<string, any>>(
  tableName: string,
  options: FetchSupabaseTableOptions = {}
): Promise<T[]> => {
  const config = getSupabaseConfig()

  if (!config) {
    throw new Error('Supabase is not configured. Missing VITE_SUPABASE_URL or VITE_SUPABASE_ANON_KEY.')
  }

  if (!tableName) {
    throw new Error('Supabase table name is required')
  }

  const { orderBy = 'created_at', ascending = false, limit, signal } = options

  const endpoint = new URL(`${config.url}/rest/v1/${tableName}`)
  endpoint.searchParams.set('select', '*')

  if (orderBy) {
    endpoint.searchParams.set('order', `${orderBy}.${ascending ? 'asc' : 'desc'}`)
  }

  const headers: Record<string, string> = {
    apikey: config.apiKey,
    Authorization: `Bearer ${config.apiKey}`,
    Accept: 'application/json'
  }

  if (typeof limit === 'number' && limit > 0) {
    headers['Range'] = `0-${limit - 1}`
    headers['Range-Unit'] = 'items'
  }

  const response = await fetch(endpoint.toString(), {
    headers,
    signal
  })

  if (!response.ok) {
    let errorMessage = `Supabase request failed with status ${response.status}`

    try {
      const body = await response.json()
      if (body?.message) {
        errorMessage = body.message
      }
    } catch (parseError) {
      const text = await response.text()
      if (text) {
        errorMessage = text
      }
    }

    throw new Error(errorMessage)
  }

  const data = (await response.json()) as T[]
  return data ?? []
}
