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

<<<<<<< HEAD
const TRUNCATE = 'block overflow-hidden text-ellipsis whitespace-nowrap'

=======
>>>>>>> ee334457d998ea07aca69a5c1f4838a95bfd3354
export default function DataTable<TData, TValue>({ columns, data }: DataTableProps<TData, TValue>) {
  const [columnSizing, setColumnSizing] = useState<ColumnSizingState>({})
  const [columnSizingInfo, setColumnSizingInfo] = useState<ColumnSizingInfoState>({} as any)

  useEffect(() => {
<<<<<<< HEAD
=======
    // While actively resizing, prevent accidental text selection
>>>>>>> ee334457d998ea07aca69a5c1f4838a95bfd3354
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
<<<<<<< HEAD
    columnResizeMode: 'onChange',
=======
    columnResizeMode: 'onChange', // use 'onEnd' if you prefer applying size on mouseup
>>>>>>> ee334457d998ea07aca69a5c1f4838a95bfd3354
    onColumnSizingChange: setColumnSizing,
    onColumnSizingInfoChange: setColumnSizingInfo,
    state: { columnSizing, columnSizingInfo }
  })

  return (
    <div className="overflow-auto rounded-md border">
<<<<<<< HEAD
      {/* fixed layout keeps column widths stable while resizing */}
=======
>>>>>>> ee334457d998ea07aca69a5c1f4838a95bfd3354
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
<<<<<<< HEAD
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
=======
                    className="whitespace-nowrap"
                  >
                    {header.isPlaceholder ? null : (
                      <div className="pr-3">
                        {flexRender(header.column.columnDef.header, header.getContext())}
                      </div>
                    )}

                    {/* BIG hit area (easy to grab) */}
>>>>>>> ee334457d998ea07aca69a5c1f4838a95bfd3354
                    <div
                      onMouseDown={header.getResizeHandler()}
                      onTouchStart={header.getResizeHandler()}
                      onDoubleClick={() => header.column.resetSize()}
<<<<<<< HEAD
                      className="absolute top-0 right-0 -mr-2 h-full w-[16px] cursor-col-resize touch-none select-none z-50"
=======
                      className={[
                        // large invisible hitbox that slightly overflows right side
                        'absolute top-0 right-0 -mr-2 h-full w-[14px]',
                        'cursor-col-resize touch-none select-none z-50'
                      ].join(' ')}
>>>>>>> ee334457d998ea07aca69a5c1f4838a95bfd3354
                      style={{ touchAction: 'none' }}
                      aria-label="Resize column"
                      role="separator"
                      aria-orientation="vertical"
                    >
<<<<<<< HEAD
=======
                      {/* visible thin line centered in the hitbox */}
>>>>>>> ee334457d998ea07aca69a5c1f4838a95bfd3354
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
<<<<<<< HEAD
                      className="px-3"
                    >
                      {/* Truncate wrapper for body cells too */}
                      <div className={TRUNCATE}>
                        {flexRender(cell.column.columnDef.cell, cell.getContext())}
                      </div>
=======
                      className="truncate"
                    >
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
>>>>>>> ee334457d998ea07aca69a5c1f4838a95bfd3354
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
