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

export default function DataTable<TData, TValue>({ columns, data }: DataTableProps<TData, TValue>) {
  const [columnSizing, setColumnSizing] = useState<ColumnSizingState>({})
  const [columnSizingInfo, setColumnSizingInfo] = useState<ColumnSizingInfoState>({} as any)

  useEffect(() => {
    // While actively resizing, prevent accidental text selection
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
    columnResizeMode: 'onChange', // use 'onEnd' if you prefer applying size on mouseup
    onColumnSizingChange: setColumnSizing,
    onColumnSizingInfoChange: setColumnSizingInfo,
    state: { columnSizing, columnSizingInfo }
  })

  return (
    <div className="overflow-auto rounded-md border">
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
                    className="whitespace-nowrap"
                  >
                    {header.isPlaceholder ? null : (
                      <div className="pr-3">
                        {flexRender(header.column.columnDef.header, header.getContext())}
                      </div>
                    )}

                    {/* BIG hit area (easy to grab) */}
                    <div
                      onMouseDown={header.getResizeHandler()}
                      onTouchStart={header.getResizeHandler()}
                      onDoubleClick={() => header.column.resetSize()}
                      className={[
                        // large invisible hitbox that slightly overflows right side
                        'absolute top-0 right-0 -mr-2 h-full w-[14px]',
                        'z-50 cursor-col-resize touch-none select-none'
                      ].join(' ')}
                      style={{ touchAction: 'none' }}
                      aria-label="Resize column"
                      role="separator"
                      aria-orientation="vertical"
                    >
                      {/* visible thin line centered in the hitbox */}
                      <div
                        className={[
                          'absolute top-0 bottom-0 left-1/2 w-px -translate-x-1/2',
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
                      className="align-top"
                    >
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
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
