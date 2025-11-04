import { useEffect, useMemo, useState, useCallback } from 'react'
import {
  ColumnDef,
  flexRender,
  getCoreRowModel,
  useReactTable,
  ColumnSizingState,
  ColumnSizingInfoState
} from '@tanstack/react-table'

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow
} from '@/components/ui/Table'

interface DataTableProps<TData, TValue> {
  columns: ColumnDef<TData, TValue>[]
  data: TData[]
}

type CellPos = { row: number; col: number } | null

export default function DataTable<TData, TValue>({ columns, data }: DataTableProps<TData, TValue>) {
  const [columnSizing, setColumnSizing] = useState<ColumnSizingState>({})
  const [columnSizingInfo, setColumnSizingInfo] = useState<ColumnSizingInfoState>({} as any)
  const [selectedCell, setSelectedCell] = useState<CellPos>(null)

  // prevent text selection while resizing
  useEffect(() => {
    if ((columnSizingInfo as any)?.isResizingColumn) {
      document.body.style.userSelect = 'none'
      document.body.style.cursor = 'col-resize'
      return () => {
        document.body.style.userSelect = ''
        document.body.style.cursor = ''
      }
    }
    document.body.style.userSelect = ''
    document.body.style.cursor = ''
  }, [columnSizingInfo])

  const table = useReactTable({
    data,
    columns,
    getCoreRowModel: getCoreRowModel(),
    columnResizeMode: 'onChange',
    onColumnSizingChange: setColumnSizing,
    onColumnSizingInfoChange: setColumnSizingInfo,
    state: { columnSizing, columnSizingInfo }
  })

  const rows = table.getRowModel().rows

  // Ensure selectedCell stays in bounds if data/columns change
  useEffect(() => {
    if (!selectedCell) return
    const maxRow = rows.length - 1
    const newRow = Math.min(selectedCell.row, Math.max(0, maxRow))
    const colsForRow = rows[newRow]?.getVisibleCells() ?? []
    const maxCol = colsForRow.length - 1
    const newCol = Math.min(selectedCell.col, Math.max(0, maxCol))
    if (newRow !== selectedCell.row || newCol !== selectedCell.col) {
      setSelectedCell({ row: newRow, col: newCol })
    }
  }, [rows, selectedCell])

  const moveSelection = useCallback(
    (dir: 'up' | 'down' | 'left' | 'right') => {
      if (!rows.length) return
      setSelectedCell((prev) => {
        // default to first cell if nothing selected
        const cur = prev ?? { row: 0, col: 0 }
        let r = cur.row
        let c = cur.col

        if (dir === 'up') r = Math.max(0, r - 1)
        if (dir === 'down') r = Math.min(rows.length - 1, r + 1)

        const colsForRow = rows[r]?.getVisibleCells() ?? []
        const maxCol = Math.max(0, colsForRow.length - 1)

        if (dir === 'left') c = Math.max(0, c - 1)
        if (dir === 'right') c = Math.min(maxCol, c + 1)

        // clamp if row change altered available columns
        c = Math.min(c, maxCol)
        return { row: r, col: c }
      })
    },
    [rows]
  )

  const onKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLDivElement>) => {
      // Only arrow nav; prevent page scroll
      if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.key)) {
        e.preventDefault()
      }
      switch (e.key) {
        case 'ArrowUp':
          moveSelection('up')
          break
        case 'ArrowDown':
          moveSelection('down')
          break
        case 'ArrowLeft':
          moveSelection('left')
          break
        case 'ArrowRight':
          moveSelection('right')
          break
      }
    },
    [moveSelection]
  )

  return (
    <div
      className="overflow-auto rounded-md border outline-none"
      role="grid"
      tabIndex={0}
      onKeyDown={onKeyDown}
      aria-rowcount={rows.length}
    >
      <Table className="table-fixed select-none">
        <TableHeader>
          {table.getHeaderGroups().map((headerGroup) => (
            <TableRow key={headerGroup.id} role="row">
              {headerGroup.headers.map((header) => {
                const size = header.getSize()
                const isResizing = header.column.getIsResizing()
                return (
                  <TableHead
                    key={header.id}
                    role="columnheader"
                    style={{
                      width: size,
                      minWidth: header.column.columnDef.minSize ?? 60,
                      maxWidth: header.column.columnDef.maxSize ?? 1200,
                      position: 'relative'
                    }}
                    className="px-3"
                    title={
                      typeof header.column.columnDef.header === 'string'
                        ? header.column.columnDef.header
                        : undefined
                    }
                  >
                    {header.isPlaceholder ? null : (
                      <span className="block overflow-hidden text-ellipsis whitespace-nowrap">
                        {flexRender(header.column.columnDef.header, header.getContext())}
                      </span>
                    )}
                    <div
                      onMouseDown={header.getResizeHandler()}
                      onTouchStart={header.getResizeHandler()}
                      onDoubleClick={() => header.column.resetSize()}
                      className="absolute top-0 right-0 -mr-2 h-full w-[16px] cursor-col-resize touch-none select-none z-50"
                      style={{ touchAction: 'none' }}
                      aria-label="Resize column"
                      role="separator"
                      aria-orientation="vertical"
                    >
                      <div
                        className={[
                          'absolute top-0 bottom-0 left-1/2 -translate-x-1/2 w-px',
                          'bg-border',
                          isResizing ? 'bg-muted-foreground' : 'hover:bg-muted-foreground/60'
                        ].join(' ')}
                      />
                    </div>
                  </TableHead>
                )
              })}
            </TableRow>
          ))}
        </TableHeader>

        <TableBody>
          {rows?.length ? (
            rows.map((row) => {
              const visibleCells = row.getVisibleCells()
              return (
                <TableRow key={row.id} role="row" data-state={row.getIsSelected() && 'selected'}>
                  {visibleCells.map((cell, ci) => {
                    const size = cell.column.getSize()
                    const isSelected =
                      selectedCell?.row === row.index && selectedCell?.col === ci

                    return (
                      <TableCell
                        key={cell.id}
                        role="gridcell"
                        aria-selected={isSelected || undefined}
                        onClick={() => setSelectedCell({ row: row.index, col: ci })}
                        style={{
                          width: size,
                          minWidth: cell.column.columnDef.minSize ?? 60,
                          maxWidth: cell.column.columnDef.maxSize ?? 1200
                        }}
                        className={[
                          'px-3',
                          // visual selection (no border-radius)
                          isSelected
                            ? 'bg-accent'
                            : ''
                        ].join(' ')}
                        // allow focusing the selected cell for a11y (container holds the keyboard listener)
                        tabIndex={isSelected ? 0 : -1}
                      >
                        {flexRender(cell.column.columnDef.cell, cell.getContext())}
                      </TableCell>
                    )
                  })}
                </TableRow>
              )
            })
          ) : (
            <TableRow role="row">
              <TableCell colSpan={columns.length} className="h-24 text-center">
                No results.
              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>
    </div>
  )
}
