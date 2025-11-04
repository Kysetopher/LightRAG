import { useCallback, useEffect, useMemo, useState } from 'react'
import type { ColumnDef } from '@tanstack/react-table'
import { useTranslation } from 'react-i18next'

import { generateCsv, listCsvTemplates, type CsvTemplateInfo } from '@/api/lightrag'
import Button from '@/components/ui/Button'
import Input from '@/components/ui/Input'
import { Card } from '@/components/ui/Card'
import DataTable from '@/components/ui/DataTable'
import Textarea from '@/components/ui/Textarea'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from '@/components/ui/Select'
import { useBackendState } from '@/stores/state'

interface RowData {
  [key: string]: string
}

interface StoredCsvState {
  template: string
  customColumns: string
  limit: string
  prompt: string
  rows: RowData[]
  columns: string[]
}

const STORAGE_KEY = 'csvGeneratorState:v1'

export default function CsvGenerator() {
  const { t } = useTranslation()
  const workspace = useBackendState.use.workspace()
  const [templates, setTemplates] = useState<CsvTemplateInfo[]>([])
  const [template, setTemplate] = useState<string>('fmea')
  const [customColumns, setCustomColumns] = useState<string>('')
  const [limit, setLimit] = useState<string>('1000')
  const [prompt, setPrompt] = useState<string>('')
  const [previewRows, setPreviewRows] = useState<RowData[]>([])
  const [csvColumns, setCsvColumns] = useState<string[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [hasHydratedStorage, setHasHydratedStorage] = useState(false)

  const storageId = workspace ?? 'default'

  useEffect(() => {
    let cancelled = false
    listCsvTemplates()
      .then((items) => {
        if (cancelled) {
          return
        }
        setTemplates(items)
        setTemplate((current) => {
          if (items.some((item) => item.id === current)) {
            return current
          }
          return items[0]?.id ?? current
        })
      })
      .catch((err) => {
        if (cancelled) {
          return
        }
        console.error(err)
        setError(t('csvGenerator.errors.loadTemplates'))
      })
    return () => {
      cancelled = true
    }
  }, [t])

  useEffect(() => {
    setHasHydratedStorage(false)
    if (typeof window === 'undefined') {
      setPreviewRows([])
      setCsvColumns([])
      setHasHydratedStorage(true)
      return
    }

    try {
      const raw = window.localStorage.getItem(STORAGE_KEY)
      if (!raw) {
        setTemplate('fmea')
        setCustomColumns('')
        setLimit('1000')
        setPrompt('')
        setPreviewRows([])
        setCsvColumns([])
        setError(null)
        setHasHydratedStorage(true)
        return
      }

      const parsed = JSON.parse(raw) as Record<string, StoredCsvState | undefined>
      const storedState = parsed?.[storageId]

      if (!storedState) {
        setTemplate('fmea')
        setCustomColumns('')
        setLimit('1000')
        setPrompt('')
        setPreviewRows([])
        setCsvColumns([])
        setError(null)
        setHasHydratedStorage(true)
        return
      }

      setTemplate(storedState.template ?? 'fmea')
      setCustomColumns(storedState.customColumns ?? '')
      setLimit(storedState.limit ?? '1000')
      setPrompt(storedState.prompt ?? '')
      setPreviewRows(Array.isArray(storedState.rows) ? storedState.rows : [])
      setCsvColumns(Array.isArray(storedState.columns) ? storedState.columns : [])
      setError(null)
    } catch (err) {
      console.error('Failed to restore CSV generator state', err)
      setTemplate('fmea')
      setCustomColumns('')
      setLimit('1000')
      setPrompt('')
      setPreviewRows([])
      setCsvColumns([])
      setError(null)
    } finally {
      setHasHydratedStorage(true)
    }
  }, [storageId])

  useEffect(() => {
    if (!hasHydratedStorage || typeof window === 'undefined') {
      return
    }

    try {
      const raw = window.localStorage.getItem(STORAGE_KEY)
      const parsed = raw ? (JSON.parse(raw) as Record<string, StoredCsvState>) : {}
      parsed[storageId] = {
        template,
        customColumns,
        limit,
        prompt,
        rows: previewRows,
        columns: csvColumns
      }
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(parsed))
    } catch (err) {
      console.error('Failed to persist CSV generator state', err)
    }
  }, [
    storageId,
    hasHydratedStorage,
    template,
    customColumns,
    limit,
    prompt,
    previewRows,
    csvColumns
  ])

  const columns = useMemo(() => {
    if (template === 'custom') {
      return customColumns
        .split(',')
        .map((value) => value.trim())
        .filter(Boolean)
    }
    return templates.find((item) => item.id === template)?.columns ?? []
  }, [customColumns, template, templates])

  const activeColumns = csvColumns.length ? csvColumns : columns

  const handleCellChange = useCallback(
    (rowIndex: number, columnId: string, value: string) => {
      setPreviewRows((previous) => {
        if (!previous[rowIndex]) {
          return previous
        }
        const updated = [...previous]
        updated[rowIndex] = {
          ...updated[rowIndex],
          [columnId]: value
        }
        return updated
      })
    },
    [setPreviewRows]
  )

  const tableColumns = useMemo<ColumnDef<RowData>[]>(() => {
    return activeColumns.map((columnName) => ({
      header: columnName,
      accessorKey: columnName,
      cell: ({ row, column, getValue }) => (
        <Input
          value={(getValue() as string) ?? ''}
          onChange={(event) => handleCellChange(row.index, column.id as string, event.target.value)}
          className="h-9 w-full"
        />
      )
    }))
  }, [activeColumns, handleCellChange])

  const limitValue = useMemo(() => {
    const numeric = Number(limit)
    return Number.isFinite(numeric) && numeric > 0 ? Math.floor(numeric) : 1
  }, [limit])

  const disablePreview = template === 'custom' && columns.length === 0
  const disableDownload =
    template === 'custom' &&
    columns.length === 0 &&
    previewRows.length === 0 &&
    csvColumns.length === 0

  const parseCsvText = useCallback((text: string) => {
    const lines = text.split(/\r?\n/).filter((line) => line.length > 0)
    if (!lines.length) {
      return { headers: [] as string[], rows: [] as RowData[] }
    }

    const [headerLine, ...dataLines] = lines
    const headers = headerLine.split(',')
    const rows = dataLines.map<RowData>((line) => {
      const values = line.split(',')
      return headers.reduce<RowData>((acc, key, index) => {
        acc[key] = values[index] ?? ''
        return acc
      }, {})
    })

    return { headers, rows }
  }, [])

  const createCsvContent = useCallback((headers: string[], rows: RowData[]) => {
    if (!headers.length) {
      return ''
    }

    const encodeValue = (value: string | undefined) => {
      const raw = value ?? ''
      if (/[",\n\r]/.test(raw)) {
        return `"${raw.replace(/"/g, '""')}"`
      }
      return raw
    }

    const headerLine = headers.map((header) => encodeValue(header)).join(',')
    const lines = rows.map((row) =>
      headers.map((header) => encodeValue(row[header] ?? '')).join(',')
    )
    return [headerLine, ...lines].join('\n')
  }, [])

  const downloadFromRows = useCallback(
    (headers: string[], rows: RowData[]) => {
      if (!headers.length) {
        setError(t('csvGenerator.errors.downloadFailed'))
        return
      }

      const csvContent = createCsvContent(headers, rows)
      const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' })
      const url = URL.createObjectURL(blob)
      const anchor = document.createElement('a')
      anchor.href = url
      anchor.download = `${template}.csv`
      anchor.click()
      URL.revokeObjectURL(url)
    },
    [createCsvContent, setError, t, template]
  )

  const handleTemplateChange = useCallback(
    (value: string) => {
      setTemplate(value)
      setPreviewRows([])
      setCsvColumns([])
      setError(null)
    },
    [setTemplate, setPreviewRows, setCsvColumns, setError]
  )

  const handleCustomColumnsChange = useCallback(
    (value: string) => {
      setCustomColumns(value)
      setPreviewRows([])
      setCsvColumns([])
      setError(null)
    },
    [setCustomColumns, setPreviewRows, setCsvColumns, setError]
  )

  const handlePromptChange = useCallback(
    (value: string) => {
      setPrompt(value)
      setPreviewRows([])
      setCsvColumns([])
      setError(null)
    },
    [setPrompt, setPreviewRows, setCsvColumns, setError]
  )

  const handlePreview = async () => {
    setLoading(true)
    setError(null)
    try {
      const blob = await generateCsv({
        workspace: workspace || undefined,
        template,
        prompt: prompt.trim() ? prompt : undefined,
        columns: template === 'custom' ? columns : undefined,
        limit: limitValue
      })
      const text = await blob.text()
      const { headers, rows } = parseCsvText(text)
      setCsvColumns(headers)
      setPreviewRows(rows)
    } catch (err) {
      console.error(err)
      setPreviewRows([])
      setCsvColumns([])
      setError(t('csvGenerator.errors.previewFailed'))
    } finally {
      setLoading(false)
    }
  }

  const handleDownload = async () => {
    setError(null)
    const headers = csvColumns.length ? csvColumns : columns

    if (previewRows.length && headers.length) {
      downloadFromRows(headers, previewRows)
      return
    }

    if (template === 'custom' && columns.length === 0) {
      setError(t('csvGenerator.errors.downloadFailed'))
      return
    }

    setLoading(true)
    try {
      const blob = await generateCsv({
        workspace: workspace || undefined,
        template,
        prompt: prompt.trim() ? prompt : undefined,
        columns: template === 'custom' ? columns : undefined,
        limit: limitValue
      })
      const text = await blob.text()
      const { headers: fetchedHeaders, rows } = parseCsvText(text)
      const resolvedHeaders = fetchedHeaders.length ? fetchedHeaders : headers
      setCsvColumns(resolvedHeaders)
      setPreviewRows(rows)
      downloadFromRows(resolvedHeaders, rows)
    } catch (err) {
      console.error(err)
      setError(t('csvGenerator.errors.downloadFailed'))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="grid gap-4 p-4">
      {/* Top split card (grows to fill remaining height) */}
      <Card className="p-4">
        <div className="grid h-full grid-cols-1 gap-4 md:grid-cols-3">
          {/* Left pane: controls + buttons + fixed error row */}
          <div className="grid min-h-0 grid-rows-[auto,auto,auto,1fr,auto,auto] gap-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">{t('csvGenerator.templateLabel')}</label>
              <Select value={template} onValueChange={handleTemplateChange}>
                <SelectTrigger>
                  <SelectValue placeholder={t('csvGenerator.templatePlaceholder')} />
                </SelectTrigger>
                <SelectContent>
                  {templates.map((item) => (
                    <SelectItem key={item.id} value={item.id}>
                      {item.id}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {template === 'custom' && (
              <div className="space-y-2">
                <label className="text-sm font-medium">
                  {t('csvGenerator.customColumnsLabel')}
                </label>
                <Input
                  placeholder={t('csvGenerator.customColumnsPlaceholder')}
                  value={customColumns}
                  onChange={(e) => handleCustomColumnsChange(e.target.value)}
                />
              </div>
            )}

            <div className="space-y-2">
              <label className="text-sm font-medium">{t('csvGenerator.limitLabel')}</label>
              <Input
                type="number"
                min={1}
                value={limit}
                onChange={(e) => setLimit(e.target.value)}
              />
            </div>

            {/* spacer fills extra space so buttons stay at bottom */}
            <div />

            {/* Buttons: left-aligned, pinned to bottom of left pane */}
            <div className="mt-auto flex flex-wrap gap-2">
              <Button onClick={handlePreview} disabled={loading || disablePreview}>
                {t('csvGenerator.previewButton')}
              </Button>
              <Button
                variant="outline"
                onClick={handleDownload}
                disabled={loading || disableDownload}
              >
                {t('csvGenerator.downloadButton')}
              </Button>
            </div>

            {/* Fixed-height error slot prevents layout shift */}
            <p aria-live="polite" className="text-destructive h-5 text-sm">
              {error || '\u00A0'}
            </p>
          </div>

          {/* Right pane: label + stretchy Textarea */}
          <div className="col-span-2 flex min-h-0 flex-col">
            <label className="mb-2 text-sm font-medium">{t('csvGenerator.promptLabel')}</label>
            <Textarea
              placeholder={t('csvGenerator.promptPlaceholder') ?? undefined}
              value={prompt}
              onChange={(e) => handlePromptChange(e.target.value)}
              className="resize-vertical h-auto min-h-40 flex-1"
            />
          </div>
        </div>
      </Card>

      {/* Bottom card: data table (auto height) */}
      <div className="overflow-auto">
        {loading ? (
          <Card className="text-muted-foreground p-6 text-sm">
            {t('csvGenerator.loadingMessage')}
          </Card>
        ) : previewRows.length ? (
          <DataTable columns={tableColumns} data={previewRows} />
        ) : (
          <Card className="text-muted-foreground p-6 text-sm">{t('csvGenerator.emptyState')}</Card>
        )}
      </div>
    </div>
  )
}
