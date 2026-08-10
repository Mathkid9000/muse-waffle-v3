import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useStore } from '../../store'
import { useAuth } from '../../contexts/AuthContext'
import { formatDate, getScore, saveScore, setScoreName, type ScoreRow } from '../../lib/scores'
import { countInstruments } from '../../hooks/useVerovio'
import styles from './Toolbar.module.css'
import Avatar from '../Avatar/Avatar'

export default function Toolbar() {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const navigate = useNavigate()
  const { user, profile } = useAuth()
  const getXml = useStore((s) => s.getXml)
  const currentXml = useStore((s) => s.currentXml)
  const scoreId = useStore((s) => s.scoreId)
  const setScoreId = useStore((s) => s.setScoreId)
  const zoom = useStore((s) => s.zoom)
  const setXml = useStore((s) => s.setXml)
  const setScoreData = useStore((s) => s.setScoreData)
  const setError = useStore((s) => s.setError)
  const zoomTo = useStore((s) => s.zoomTo)
  const resetCamera = useStore((s) => s.resetCamera)
  const redrawScore = useStore((s) => s.redrawScore)
  const viewportW = useStore((s) => s.viewportW)
  const viewportH = useStore((s) => s.viewportH)
  const tabs = useStore((s) => s.tabs)
  const activeTab = useStore((s) => s.activeTab)
  const setActiveTab = useStore((s) => s.setActiveTab)

  const [ isFileDdOpen, setIsFileDdOpen ] = useState<boolean>(false)
  const [ isViewDdOpen, setIsViewDdOpen ] = useState<boolean>(false)
  const [ isExitDdOpen, setIsExitDdOpen ] = useState<boolean>(false)
  const [ scoreRow, setScoreRow ] = useState<ScoreRow | null>(null)

  // Toolbar zoom anchors at the viewport center (the wheel anchors at the cursor).
  const zoomStep = (delta: number) => zoomTo(zoom + delta, viewportW / 2, viewportH / 2)

  async function loadScoreXml(xml: string) {
    const numInstruments = await countInstruments(xml)
    setScoreData({ numInstruments })
    setXml(xml, 'Main')
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
    if (!currentXml || !user) return
    const id = await saveScore(user.id, currentXml, scoreId)
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
    if (!getXml('Main')) return
    const blob = new Blob([getXml('Main')!], { type: 'application/xml' })
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
        <button className={styles.dropdownBtn} onClick={() => setIsViewDdOpen(true)} onMouseLeave={() => setIsViewDdOpen(false)}>
          View
          {isViewDdOpen && (
            <ul className={styles.dropdownContainer}>
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
              <li className={styles.dropdownItem} onClick={async () => { await handleSave(); navigate('/dashboard') }}>Save and Exit</li>
              <li className={styles.dropdownItem} onClick={() => navigate('/dashboard')}>Exit without Saving</li>
            </ul>
          )}
        </button>
        {scoreRow && (<p className={styles.lastSaved}>Last Saved {formatDate(scoreRow.updated_at)}</p>)}
      </div>
      <div className={styles.tabs}>
        {tabs.map(t => (
          <button className={t === activeTab ? styles.activeTab : styles.tab}
            onClick={() => setActiveTab(t)}>{t}</button>
        ))}
      </div>
    </header>
  )
}
