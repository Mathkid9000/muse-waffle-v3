import { act, useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useStore } from '../../store'
import { useAuth } from '../../contexts/AuthContext'
import { formatDate, getScore, saveScore, setScoreName, type ScoreRow } from '../../lib/scores'
import { countInstruments } from '../../hooks/useVerovio'
import styles from './Toolbar.module.css'
import Avatar from '../Avatar/Avatar'
import { tabTemplate } from '../../types'

export default function Toolbar() {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const navigate = useNavigate()
  const { user, profile } = useAuth()
  const getXml = useStore((s) => s.getXml)
  const scoreId = useStore((s) => s.scoreId)
  const setScoreId = useStore((s) => s.setScoreId)
  const zoom = useStore((s) => s.zoom)
  const setScoreData = useStore((s) => s.setScoreData)
  const setError = useStore((s) => s.setError)
  const zoomTo = useStore((s) => s.zoomTo)
  const resetCamera = useStore((s) => s.resetCamera)
  const redrawScore = useStore((s) => s.redrawScore)
  const viewportW = useStore((s) => s.viewportW)
  const viewportH = useStore((s) => s.viewportH)
  const tabs = useStore((s) => s.tabs)
  const setTabs = useStore((s) => s.setTabs)
  const addNewTab = useStore((s) => s.addNewTab)
  const deleteTab = useStore((s) => s.deleteTab)
  const renameTab = useStore((s) => s.renameTab)
  const activeTab = useStore((s) => s.activeTab)
  const setActiveTab = useStore((s) => s.setActiveTab)
  const textView = useStore((s) => s.textView)
  const toggleTextView = useStore((s) => s.toggleTextView)
  const checkpoint = useStore((s) => s.checkpoint)
  const undo = useStore((s) => s.undo)
  const redo = useStore((s) => s.redo)
  const undoLabel = useStore((s) => s.past.at(-1)?.label)
  const redoLabel = useStore((s) => s.future.at(-1)?.label)

  const chats = useStore((s) => s.chats)
  const deleteChat = useStore((s) => s.deleteChat)

  const [ isFileDdOpen, setIsFileDdOpen ] = useState<boolean>(false)
  const [ isEditDdOpen, setIsEditDdOpen ] = useState<boolean>(false)
  const [ isViewDdOpen, setIsViewDdOpen ] = useState<boolean>(false)
  const [ isTemplateDdOpen, setIsTemplateDdOpen ] = useState<boolean>(false)
  const [ isExitDdOpen, setIsExitDdOpen ] = useState<boolean>(false)
  const [ scoreRow, setScoreRow ] = useState<ScoreRow | null>(null)
  const [ showEditingTabName, setShowEditingTabName ] = useState<boolean>(false)

  // Toolbar zoom anchors at the viewport center (the wheel anchors at the cursor).
  const zoomStep = (delta: number) => zoomTo(zoom + delta, viewportW / 2, viewportH / 2)

  async function loadScoreXml(xml: string) {
    const numInstruments = await countInstruments(xml)
    setScoreData({ numInstruments })
    // A freshly imported file replaces the whole document — drop any tabs left over
    // from whatever was open before instead of appending to them.
    checkpoint('Import')
    const tab = addNewTab('Imported Score', { name: "Imported Score", content: xml })
    setActiveTab(tab)
  }

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    if (file.name.toLowerCase().endsWith('.mxl')) {
      // MXL is a ZIP archive — read as binary and pass base64 to Verovio's ZIP loader
      reader.onload = (ev) => {
        const buf = ev.target?.result as ArrayBuffer
        const bytes = new Uint8Array(buf)
        let binary = ''
        for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i])
        loadScoreXml('__MXL__' + btoa(binary))
      }
      reader.onerror = () => setError('Failed to read file.')
      reader.readAsArrayBuffer(file)
    } else {
      reader.onload = (ev) => {
        if (typeof ev.target?.result === 'string') loadScoreXml(ev.target.result)
      }
      reader.onerror = () => setError('Failed to read file.')
      reader.readAsText(file)
    }
    e.target.value = ''
  }

  async function handleSave() {
    if (!user) return
    const id = await saveScore(user.id, tabs, scoreId, useStore.getState().chats, activeTab)
    // The preview render leaves the shared toolkit on a portrait page — put the editor's back.
    redrawScore()
    if (!id) return setError('Failed to save score.')
    if (id !== scoreId) {
      setScoreId(id)
      navigate(`/editor/${id}`, { replace: true })
    }
    const newScoreRow = await getScore(id)
    setScoreRow(newScoreRow)
  }

  // Rebound every render so the handler always closes over the current xml and score id.
  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 's') {
        e.preventDefault()
        handleSave()
      }
      // Text fields (chat box, score name, a tab mid-rename) keep their own native undo.
      const target = e.target as HTMLElement
      if (target.isContentEditable || target.tagName === 'INPUT' || target.tagName === 'TEXTAREA') return
      const key = e.key.toLowerCase()
      if ((e.metaKey || e.ctrlKey) && (key === 'y' || (key === 'z' && e.shiftKey))) {
        e.preventDefault()
        redo()
      } else if ((e.metaKey || e.ctrlKey) && key === 'z') {
        e.preventDefault()
        undo()
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  })

  useEffect(() => {
    const func = async () => {
      if (scoreId == null) {
        setScoreRow(null)
        return
      }
      const row = await getScore(scoreId)
      setScoreRow(row)
    }
    func()
  }, [scoreId])

  function handleDownload() {
    if (!getXml(activeTab)) return
    const blob = new Blob([getXml(activeTab)!], { type: 'application/xml' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'score.musicxml'
    a.click()
    URL.revokeObjectURL(url)
  }

  function handleNameChange(e: EventTarget, event?: () => void) {
    if (event) { 
      try {
        event()
      }
      catch {}
      e.blur()
    }

    if (!scoreId) return
    const newName = (e as HTMLElement).textContent
    setScoreName(scoreId, newName)
    console.log('Name changed to ', newName)
  }

  function createNewTemplate(e: React.MouseEvent) {
    e.stopPropagation()
    setIsTemplateDdOpen(false)
    // A template tab starts as an empty item list — that's what marks it as a template.
    checkpoint('Create Tab')
    const tabName = addNewTab('New Template', { content: [] })
    setActiveTab(tabName)
  }

  function renameCurrentTemplate(e: React.MouseEvent) {
    if (activeTab === 'Main') return
    const button = document.getElementsByClassName(styles.activeTab)[0] as HTMLElement
    if (!button) return
    e.stopPropagation()
    setIsTemplateDdOpen(false)
    button.focus()
    const name = activeTab
    button.contentEditable = true
    setShowEditingTabName(true)
    
    let renamed = false

    // Attached fresh on every Rename click — without this, a listener from a previous
    // (possibly abandoned) rename stays on the DOM node React reuses for this tab, and
    // both fire on the next Enter: the stale one re-runs the rename against its old,
    // pre-render tab list and collides with the tab the live one just created.
    function cleanup() {
      button.removeEventListener('blur', onBlur)
      button.removeEventListener('keydown', onKeyDown)
      setShowEditingTabName(false)
    }

    function onBlur() {
      button.contentEditable = false
      button.style.outline = "none"
      if (!renamed) button.textContent = name
      cleanup()
    }

    function onKeyDown(e: KeyboardEvent) {
      if (e.key !== 'Enter') return
      e.preventDefault()

      const newName = button.textContent
      renamed = true
      button.contentEditable = false
      button.style.outline = "none"
      cleanup()

      // The tab carries its own content, so renaming is a field edit — nothing to move.
      checkpoint('Rename Tab')
      setActiveTab(renameTab(name, newName ?? name))
    }

    button.addEventListener('blur', onBlur)
    button.addEventListener('keydown', onKeyDown)
  }

  // A template is data, so its text form is just its JSON — ScoreViewer prints it in place
  // of the item list while this is on.
  function viewAsText() {
    setIsViewDdOpen(false)
    toggleTextView()
  }

  function deleteCurrentTab(e: Event) {
    if (activeTab === 'Main') return
    e.stopPropagation()
    setIsTemplateDdOpen(false)
    checkpoint('Delete Tab')
    deleteTab(activeTab)
    setActiveTab('Main')
  }

  return (
    <header className={styles.toolbar}>
      <span className={styles.brand}>
        {/* {!noChanges && (<span className={styles.unsavedChanges}>*</span>)} */}
        <span className={scoreId ? `${styles.brand} ${styles.editable}` : `${styles.brand}`} contentEditable spellcheck={"false"} 
          onKeyDown={(e) => e.key === 'Enter' ? handleNameChange(e.target, e.preventDefault) : {}} 
          onBlur={(e) => handleNameChange(e.target)}>
            {!scoreRow ? 'Unsaved Score' : scoreRow.name}
        </span>
      </span>
      <div className={styles.avatarContainer}>
        <Avatar profile={profile} />
      </div>
      <input
        ref={fileInputRef}
        type="file"
        accept=".xml,.musicxml,.mxl"
        style={{ display: 'none' }}
        onChange={handleFileChange}
      />
      <div className={styles.dropdownGrid}>
        <button className={styles.dropdownBtn} onClick={() => setIsFileDdOpen(true)} onMouseLeave={() => setIsFileDdOpen(false)}>
          File
          {isFileDdOpen && (
            <ul className={styles.dropdownContainer}>
              <li className={styles.dropdownItem} onClick={() => fileInputRef.current?.click()}>Import File</li>
              <li className={styles.dropdownItem} onClick={handleSave}>Save</li>
              <li className={styles.dropdownItem} onClick={handleDownload}>Export</li>
            </ul>
          )}
        </button>
        <button className={styles.dropdownBtn} onClick={() => setIsEditDdOpen(true)} onMouseLeave={() => setIsEditDdOpen(false)}>
          Edit
          {isEditDdOpen && (
            <ul className={styles.dropdownContainer}>
              <li className={!undoLabel ? `${styles.dropdownItem} ${styles.disabled}` : `${styles.dropdownItem}`} onClick={undo}>Undo{undoLabel && ` ${undoLabel}`}</li>
              <li className={!redoLabel ? `${styles.dropdownItem} ${styles.disabled}` : `${styles.dropdownItem}`} onClick={redo}>Redo{redoLabel && ` ${redoLabel}`}</li>
            </ul>
          )}
        </button>
        <button className={styles.dropdownBtn} onClick={() => setIsTemplateDdOpen(true)} onMouseLeave={() => setIsTemplateDdOpen(false)}>
          Tabs
          {isTemplateDdOpen && (
            <ul className={styles.dropdownContainer}>
              <li className={styles.dropdownItem} onClick={createNewTemplate}>Create New</li>
              <li className={activeTab === 'Main' ? `${styles.dropdownItem} ${styles.disabled}` : `${styles.dropdownItem}`} onClick={renameCurrentTemplate}>Rename</li>
              <li className={activeTab === 'Main' ? `${styles.dropdownItem} ${styles.disabled}` : `${styles.dropdownItem}`} onClick={deleteCurrentTab}>Delete</li>
            </ul>
          )}
        </button>
        <button className={styles.dropdownBtn} onClick={() => setIsViewDdOpen(true)} onMouseLeave={() => setIsViewDdOpen(false)}>
          View
          {isViewDdOpen && (
            <ul className={styles.dropdownContainer}>
              {tabTemplate(tabs.find(t => t.name === activeTab)) && (<li className={styles.dropdownItem} onClick={viewAsText}>{textView ? 'View as List' : 'View as Text'}</li>)}
              <li className={styles.dropdownItem} onClick={() => redrawScore()}>Redraw</li>
              <li className={zoom >= 300 ? `${styles.dropdownItem} ${styles.disabled}` : `${styles.dropdownItem}`} onClick={() => zoomStep(10)}>Zoom +</li>
              <li className={zoom <=25 ? `${styles.dropdownItem} ${styles.disabled}` : `${styles.dropdownItem}`} onClick={() => zoomStep(-10)}>Zoom -</li>
            </ul>
          )}
        </button>
        <button className={styles.dropdownBtn} onClick={() => setIsExitDdOpen(true)} onMouseLeave={() => setIsExitDdOpen(false)}>
          Exit
          {isExitDdOpen && (
            <ul className={styles.dropdownContainer}>
              <li className={styles.dropdownItem} onClick={async () => { await handleSave(); navigate('/') }}>Save and Exit</li>
              <li className={styles.dropdownItem} onClick={() => navigate('/')}>Exit without Saving</li>
            </ul>
          )}
        </button>
        {scoreRow && (<p className={styles.lastSaved}>Last Saved {formatDate(scoreRow.updated_at)}</p>)}
      </div>
      <div className={styles.tabs}>
        {tabs.map(t => (
          <button key={t.name} title={t.description}
            className={t.name === activeTab ? (showEditingTabName ? `${styles.activeTab} ${styles.editing}` : `${styles.activeTab}`) : styles.tab}
            onClick={() => setActiveTab(t.name)}>{t.name}</button>
        ))}
      </div>
    </header>
  )
}
