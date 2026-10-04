import { useEffect } from 'react'
import { useParams } from 'react-router-dom'
import Toolbar from '../components/Toolbar/Toolbar'
import Chat from '../components/Chat/Chat'
import ScoreViewer from '../components/ScoreViewer/ScoreViewer'
import { useAuth } from '../contexts/AuthContext'
import { useStore } from '../store'
import { fetchScoreXml, parseScoreDoc } from '../lib/scores'
import { countInstruments } from '../hooks/useVerovio'
import { tabXml } from '../types'
import styles from './Editor.module.css'

export default function Editor() {
  const { id } = useParams()
  const { user } = useAuth()
  // Depend on the id, not the User object: Supabase mints a fresh User on every token
  // refresh — including the one it runs when the tab regains focus — and re-running this
  // effect on that identity change is what used to blank an unsaved score on tab-out.
  const userId = user?.id

  // The store outlives navigation and rehydrates from sessionStorage, so this only has to
  // fetch a score the store isn't already holding. The scoreId check also skips the reload
  // the post-save navigate() to /editor/:id would otherwise trigger.
  useEffect(() => {
    const { scoreId, setScoreId, setScoreData, resetScore, setTabs, setActiveTab, setChats } = useStore.getState()
    if (!id) {
      // Blank editor: keep a rehydrated unsaved draft, but drop a saved score left over
      // from navigating /editor/:id -> /editor.
      if (scoreId !== null) resetScore()
      return
    }
    if (!userId || id === scoreId) return
    fetchScoreXml(userId, id).then(async (xml) => {
      if (!xml) return
      const { tabs, chats } = parseScoreDoc(xml)
      setScoreData({ numInstruments: await countInstruments(tabXml(tabs.find(t => t.name === 'Main')) ?? '') })
      // Loading a different saved score replaces the whole tab set (and chat set) with this
      // score's own; a score saved before chats were persisted comes back with a fresh chat.
      setTabs(tabs)
      setActiveTab('Main')
      setChats(chats ?? [])
      setScoreId(id)
    })
  }, [id, userId])

  return (
    <div className={styles.layout}>
      <Toolbar />
      <div className={styles.body}>
        <div className={styles.scoreColumn}>
          <main className={styles.score}>
            <ScoreViewer />
          </main>
        </div>
        <Chat />
      </div>
    </div>
  )
}
