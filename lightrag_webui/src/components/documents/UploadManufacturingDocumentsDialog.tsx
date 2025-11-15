import { useCallback, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { FileRejection } from 'react-dropzone'

import { toast } from 'sonner'

import { uploadManufacturingDocument } from '@/api/lightrag'
import Button from '@/components/ui/Button'
import FileUploader from '@/components/ui/FileUploader'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger
} from '@/components/ui/Dialog'
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue
} from '@/components/ui/Select'
import { errorMessage } from '@/lib/utils'
import { UploadIcon } from 'lucide-react'

const CSV_ACCEPT = {
  'text/csv': ['.csv']
}

interface UploadManufacturingDocumentsDialogProps {
  onDocumentsUploaded?: () => Promise<void>
}

export default function UploadManufacturingDocumentsDialog({
  onDocumentsUploaded
}: UploadManufacturingDocumentsDialogProps) {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [progresses, setProgresses] = useState<Record<string, number>>({})
  const [fileErrors, setFileErrors] = useState<Record<string, string>>({})
  const [docType, setDocType] = useState('control_plan')

  const handleRejectedFiles = useCallback(
    (rejectedFiles: FileRejection[]) => {
      rejectedFiles.forEach(({ file, errors }) => {
        let errorMsg =
          errors[0]?.message ||
          t('documentPanel.uploadDocuments.fileUploader.fileRejected', { name: file.name })

        if (errorMsg.includes('file-invalid-type')) {
          errorMsg = t('documentPanel.uploadDocuments.fileUploader.unsupportedType')
        }

        setProgresses((pre) => ({
          ...pre,
          [file.name]: 100
        }))

        setFileErrors((prev) => ({
          ...prev,
          [file.name]: errorMsg
        }))
      })
    },
    [setProgresses, setFileErrors, t]
  )

  const handleDocumentsUpload = useCallback(
    async (filesToUpload: File[]) => {
      setIsUploading(true)
      let hasSuccessfulUpload = false

      setFileErrors((prev) => {
        const newErrors = { ...prev }
        filesToUpload.forEach((file) => {
          delete newErrors[file.name]
        })
        return newErrors
      })

      const toastId = toast.loading(
        t('documentPanel.uploadDocuments.batch.uploading')
      )

      try {
        const uploadErrors: Record<string, string> = {}

        const collator = new Intl.Collator(['zh-CN', 'en'], {
          sensitivity: 'accent',
          numeric: true
        })
        const sortedFiles = [...filesToUpload].sort((a, b) =>
          collator.compare(a.name, b.name)
        )

        for (const file of sortedFiles) {
          try {
            setProgresses((pre) => ({
              ...pre,
              [file.name]: 0
            }))

            const uploadResult = await uploadManufacturingDocument(
              file,
              { docType },
              (percentCompleted) => {
                setProgresses((pre) => ({
                  ...pre,
                  [file.name]: percentCompleted
                }))
              }
            )
            if (uploadResult.status === 'duplicated') {
              uploadErrors[file.name] = t(
                'documentPanel.uploadDocuments.fileUploader.duplicateFile'
              )
              setFileErrors((prev) => ({
                ...prev,
                [file.name]: t(
                  'documentPanel.uploadDocuments.fileUploader.duplicateFile'
                )
              }))
            } else if (uploadResult.status !== 'success') {
              uploadErrors[file.name] = uploadResult.message
              setFileErrors((prev) => ({
                ...prev,
                [file.name]: uploadResult.message
              }))
            } else {
              hasSuccessfulUpload = true
            }
          } catch (err) {
            console.error(`Upload failed for ${file.name}:`, err)
            let errorMsg = errorMessage(err)

            if (err && typeof err === 'object' && 'response' in err) {
              const axiosError = err as {
                response?: { status: number; data?: { detail?: string } }
              }
              if (axiosError.response?.status === 400) {
                errorMsg = axiosError.response.data?.detail || errorMsg
              }

              setProgresses((pre) => ({
                ...pre,
                [file.name]: 100
              }))
            }

            uploadErrors[file.name] = errorMsg
            setFileErrors((prev) => ({
              ...prev,
              [file.name]: errorMsg
            }))
          }
        }

        const hasErrors = Object.keys(uploadErrors).length > 0

        if (hasErrors) {
          toast.error(t('documentPanel.uploadDocuments.batch.error'), { id: toastId })
        } else {
          toast.success(t('documentPanel.uploadDocuments.batch.success'), { id: toastId })
        }

        if (hasSuccessfulUpload && onDocumentsUploaded) {
          onDocumentsUploaded().catch((err) => {
            console.error('Error refreshing documents:', err)
          })
        }
      } catch (err) {
        console.error('Unexpected error during upload:', err)
        toast.error(
          t('documentPanel.uploadDocuments.generalError', { error: errorMessage(err) }),
          { id: toastId }
        )
      } finally {
        setIsUploading(false)
      }
    },
    [docType, setIsUploading, setProgresses, setFileErrors, t, onDocumentsUploaded]
  )

  return (
    <Dialog
      open={open}
      onOpenChange={(nextOpen) => {
        if (isUploading) {
          return
        }
        if (!nextOpen) {
          setProgresses({})
          setFileErrors({})
          setDocType('control_plan')
        }
        setOpen(nextOpen)
      }}
    >
      <DialogTrigger asChild>
        <Button
          variant="outline"
          side="bottom"
          tooltip={t('documentPanel.uploadDocuments.manufacturing.tooltip')}
          size="sm"
        >
          <UploadIcon /> {t('documentPanel.uploadDocuments.manufacturing.button')}
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-xl" onCloseAutoFocus={(e) => e.preventDefault()}>
        <DialogHeader>
          <DialogTitle>
            {t('documentPanel.uploadDocuments.manufacturing.title')}
          </DialogTitle>
          <DialogDescription>
            {t('documentPanel.uploadDocuments.manufacturing.description')}
          </DialogDescription>
        </DialogHeader>
        <div className="mt-4 space-y-2">
          <label
            htmlFor="manufacturing_doc_type"
            className="text-sm font-medium text-foreground"
          >
            {t('documentPanel.uploadDocuments.manufacturing.planTypeLabel')}
          </label>
          <Select value={docType} onValueChange={setDocType}>
            <SelectTrigger
              id="manufacturing_doc_type"
              className="h-9 w-full text-left [&>span]:truncate"
            >
              <SelectValue placeholder={t('documentPanel.uploadDocuments.manufacturing.planTypePlaceholder')} />
            </SelectTrigger>
            <SelectContent>
              <SelectGroup>
                <SelectItem value="control_plan">
                  {t('documentPanel.uploadDocuments.manufacturing.planTypeOptions.control_plan')}
                </SelectItem>
                <SelectItem value="process_flow">
                  {t('documentPanel.uploadDocuments.manufacturing.planTypeOptions.process_flow')}
                </SelectItem>
                <SelectItem value="fmea">
                  {t('documentPanel.uploadDocuments.manufacturing.planTypeOptions.fmea')}
                </SelectItem>
                <SelectItem value="ppap">
                  {t('documentPanel.uploadDocuments.manufacturing.planTypeOptions.ppap')}
                </SelectItem>
              </SelectGroup>
            </SelectContent>
          </Select>
        </div>
        <FileUploader
          accept={CSV_ACCEPT}
          maxFileCount={Infinity}
          maxSize={200 * 1024 * 1024}
          description={t('documentPanel.uploadDocuments.manufacturing.fileTypes')}
          onUpload={handleDocumentsUpload}
          onReject={handleRejectedFiles}
          progresses={progresses}
          fileErrors={fileErrors}
          disabled={isUploading}
        />
      </DialogContent>
    </Dialog>
  )
}
