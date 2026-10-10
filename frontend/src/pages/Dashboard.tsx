import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import Avatar from '../components/Avatar/Avatar'
import { claudeApi, type ClaudeKeyStatus } from '../api/client'
import { useAuth } from '../contexts/AuthContext'
import { formatDate, listScores, previewUrl, type ScoreRow } from '../lib/scores'
import { useStore } from '../store'
import styles from './Dashboard.module.css'

export default function Dashboard() {
  const { user, profile } = useAuth()
  const navigate = useNavigate()
  const [scores, setScores] = useState<ScoreRow[]>([])

  // "New score" is the one place a blank editor is meant to be blank — the persisted draft
  // has to be dropped here, since Editor now keeps whatever it rehydrates.
  function newScore() {
    useStore.getState().resetScore()
    navigate('/editor')
  }

  useEffect(() => {
    if (user) listScores(user.id).then(setScores)
  }, [user])

  // The key itself lives encrypted on the server; the page only ever learns its last 4 characters.
  const [keyStatus, setKeyStatus] = useState<ClaudeKeyStatus | null>(null)
  const [keyInput, setKeyInput] = useState('')
  const [keyError, setKeyError] = useState('')

  useEffect(() => {
    if (user) claudeApi.getKey().then(setKeyStatus).catch(e => setKeyError(e.message))
  }, [user])

  async function updateKey(request: Promise<ClaudeKeyStatus>) {
    setKeyError('')
    try {
      setKeyStatus(await request)
      setKeyInput('')
    } catch (e) {
      setKeyError((e as Error).message)
    }
  }

  return (
    <>
      <header className={styles.headerBar}>
        <h1 className={styles.museWaffleTitle}>MUSEWAFFLE</h1>
        <div className={styles.avatarContainer}>
          <Link to="/about" className={styles.aboutButton}>About</Link>
          <Avatar profile={profile} />
        </div>
      </header>
      <form className={styles.apiKeyBar} onSubmit={e => { e.preventDefault(); if (keyInput.trim()) updateKey(claudeApi.setKey(keyInput)) }}>
        {keyStatus?.has_key ? (
          <>
            <span>Claude API key saved (…{keyStatus.last4})</span>
            <button type="button" onClick={() => updateKey(claudeApi.deleteKey())}>Remove</button>
          </>
        ) : (
          <>
            <label htmlFor="claude-api-key">Claude API key</label>
            <input id="claude-api-key" type="password" autoComplete="off" spellCheck={false} placeholder="sk-ant-..."
              value={keyInput} onChange={e => setKeyInput(e.target.value)} />
            <button type="submit">Save</button>
          </>
        )}
        {keyError && <span className={styles.apiKeyError}>{keyError}</span>}
      </form>
      <div className={styles.scoreGrid}>
        <div className={styles.background}></div>
        <div className={`${styles.scoreGridItem} ${styles.newScore}`} onClick={newScore}>
          +
        </div>
        {scores.map(score => (
          <div key={score.id} className={styles.scoreGridItem} onClick={() => navigate(`/editor/${score.id}`)}>
            <img className={styles.preview} src={previewUrl(score.user_id, score.id, score.updated_at)} />
            <div className={styles.scoreInfo}>
              <h2 className={styles.scoreName}>{score.name}</h2>
              <p className={styles.scoreDate}>{formatDate(score.updated_at)}</p>
            </div>
          </div>
        ))}
      </div>
    </>
  )
}
