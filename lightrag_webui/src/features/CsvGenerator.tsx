import { useEffect, useMemo, useState } from 'react'
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

export default function CsvGenerator() {
  const { t } = useTranslation()
  const workspace = useBackendState.use.workspace()
  const [templates, setTemplates] = useState<CsvTemplateInfo[]>([])
  const [template, setTemplate] = useState<string>('fmea')
  const [customColumns, setCustomColumns] = useState<string>('')
  const [limit, setLimit] = useState<string>('1000')
  const [prompt, setPrompt] = useState<string>('')
  const [previewRows, setPreviewRows] = useState<RowData[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

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

  const columns = useMemo(() => {
    if (template === 'custom') {
      return customColumns
        .split(',')
        .map((value) => value.trim())
        .filter(Boolean)
    }
    return templates.find((item) => item.id === template)?.columns ?? []
  }, [customColumns, template, templates])

  const tableColumns = useMemo<ColumnDef<RowData>[]>(() => {
    return columns.map((column) => ({
      header: column,
      accessorKey: column
    }))
  }, [columns])

  const limitValue = useMemo(() => {
    const numeric = Number(limit)
    return Number.isFinite(numeric) && numeric > 0 ? Math.floor(numeric) : 1
  }, [limit])

  const disableCustomActions = template === 'custom' && columns.length === 0

  useEffect(() => {
    setPreviewRows([])
    setError(null)
  }, [template, customColumns, prompt])

  const handlePreview = async () => {
    setLoading(true)
    setError(null)
    try {
      const blob = await generateCsv({
        workspace: workspace || undefined,
        template,
        prompt: prompt.trim() ? prompt : undefined,
        columns: template === 'custom' ? columns : undefined,
        limit: 50
      })
      const text = await blob.text()
      const [headerLine, ...lines] = text.split(/\r?\n/).filter(Boolean)
      if (!headerLine) {
        setPreviewRows([])
        return
      }
      const headers = headerLine.split(',')
      const rows = lines.map<RowData>((line) => {
        const values = line.split(',')
        return headers.reduce<RowData>((acc, key, index) => {
          acc[key] = values[index] ?? ''
          return acc
        }, {})
      })
      setPreviewRows(rows)
    } catch (err) {
      console.error(err)
      setPreviewRows([])
      setError(t('csvGenerator.errors.previewFailed'))
    } finally {
      setLoading(false)
    }
  }

  const handleDownload = async () => {
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
      const url = URL.createObjectURL(blob)
      const anchor = document.createElement('a')
      anchor.href = url
      anchor.download = `${template}.csv`
      anchor.click()
      URL.revokeObjectURL(url)
    } catch (err) {
      console.error(err)
      setError(t('csvGenerator.errors.downloadFailed'))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className=" grid gap-4 p-4 ">

      {/* Top split card (grows to fill remaining height) */}
      <Card className="p-4 row-start-2 row-end-3 h-full">
        <div className="grid h-full grid-cols-1 gap-4 md:grid-cols-3">
          {/* Left pane: controls + buttons + fixed error row */}
          <div className="min-h-0 grid grid-rows-[auto,auto,auto,1fr,auto,auto] gap-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">{t('csvGenerator.templateLabel')}</label>
              <Select value={template} onValueChange={setTemplate}>
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
                <label className="text-sm font-medium">{t('csvGenerator.customColumnsLabel')}</label>
                <Input
                  placeholder={t('csvGenerator.customColumnsPlaceholder')}
                  value={customColumns}
                  onChange={(e) => setCustomColumns(e.target.value)}
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
              <Button onClick={handlePreview} disabled={loading || disableCustomActions}>
                {t('csvGenerator.previewButton')}
              </Button>
              <Button
                variant="outline"
                onClick={handleDownload}
                disabled={loading || disableCustomActions}
              >
                {t('csvGenerator.downloadButton')}
              </Button>
            </div>

            {/* Fixed-height error slot prevents layout shift */}
            <p
              aria-live="polite"
              className="h-5 text-sm text-destructive"
            >
              {error || '\u00A0'}
            </p>
          </div>

          {/* Right pane: label + stretchy Textarea */}
          <div className="md:col-span-2 flex min-h-0 flex-col">
            <label className="mb-2 text-sm font-medium">{t('csvGenerator.promptLabel')}</label>
            <Textarea
              placeholder={t('csvGenerator.promptPlaceholder') ?? undefined}
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              className="flex-1 h-auto min-h-40 resize-vertical"
            />
          </div>
        </div>
      </Card>

      {/* Bottom card: data table (auto height) */}
      <Card className="p-4 row-start-3 row-end-4">
        {loading ? (
          <div className="p-6 text-sm text-muted-foreground">
            {t('csvGenerator.loadingMessage')}
          </div>
        ) : previewRows.length ? (
          <DataTable columns={tableColumns} data={previewRows} />
        ) : (
          <div className="p-6 text-sm text-muted-foreground">
            {t('csvGenerator.emptyState')}
          </div>
        )}
      </Card>
    </div>
  )
}
