import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import Avatar from '../components/Avatar/Avatar'
import { useAuth } from '../contexts/AuthContext'
import { listScores, previewUrl, type ScoreRow } from '../lib/scores'
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

  return (
    <>
      <header className={styles.headerBar}>
        <h1 className={styles.museWaffleTitle}>MUSEWAFFLE</h1>
        <div className={styles.avatarContainer}>
          <Avatar profile={profile} />
        </div>
      </header>
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
              <p className={styles.scoreDate}>{new Intl.DateTimeFormat('en-US').format(Date.parse(score.updated_at))}</p>
            </div>
          </div>
        ))}
      </div>
    </>
  )
}
