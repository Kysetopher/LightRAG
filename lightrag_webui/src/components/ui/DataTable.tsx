import { useEffect, useState } from 'react'
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

const TRUNCATE = 'block overflow-hidden text-ellipsis whitespace-nowrap'

export default function DataTable<TData, TValue>({ columns, data }: DataTableProps<TData, TValue>) {
  const [columnSizing, setColumnSizing] = useState<ColumnSizingState>({})
  const [columnSizingInfo, setColumnSizingInfo] = useState<ColumnSizingInfoState>({} as any)

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

  return (
    <div className="overflow-auto rounded-md border">
      {/* fixed layout keeps column widths stable while resizing */}
      <Table className="table-fixed">
        <TableHeader>
          {table.getHeaderGroups().map((headerGroup) => (
            <TableRow key={headerGroup.id}>
              {headerGroup.headers.map((header) => {
                const size = header.getSize()
                const isResizing = header.column.getIsResizing()

                return (
                  <TableHead
                    key={header.id}
                    style={{
                      width: size,
                      minWidth: header.column.columnDef.minSize ?? 60,
                      maxWidth: header.column.columnDef.maxSize ?? 1200,
                      position: 'relative'
                    }}
                    // unify padding with body cells
                    className="px-3"
                  >
                    {header.isPlaceholder ? null : (
                      // The actual text wrapper that truncates
                      <span
                        className={TRUNCATE}
                        title={
                          typeof header.column.columnDef.header === 'string'
                            ? header.column.columnDef.header
                            : undefined
                        }
                      >
                        {flexRender(header.column.columnDef.header, header.getContext())}
                      </span>
                    )}

                    {/* Big hit area; doesn't affect layout */}
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
          {table.getRowModel().rows?.length ? (
            table.getRowModel().rows.map((row) => (
              <TableRow key={row.id} data-state={row.getIsSelected() && 'selected'}>
                {row.getVisibleCells().map((cell) => {
                  const size = cell.column.getSize()
                  return (
                    <TableCell
                      key={cell.id}
                      style={{
                        width: size,
                        minWidth: cell.column.columnDef.minSize ?? 60,
                        maxWidth: cell.column.columnDef.maxSize ?? 1200
                      }}
                      className="px-3"
                    >
                      {/* Truncate wrapper for body cells too */}
                      <div className={TRUNCATE}>
                        {flexRender(cell.column.columnDef.cell, cell.getContext())}
                      </div>
                    </TableCell>
                  )
                })}
              </TableRow>
            ))
          ) : (
            <TableRow>
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
