import Toolbar from './components/Toolbar/Toolbar'
import ScoreViewer from './components/ScoreViewer/ScoreViewer'
import AIPanel from './components/AIPanel/AIPanel'
import PlaybackBar from './components/PlaybackBar/PlaybackBar'
import styles from './App.module.css'

export default function App() {
  return (
    <div className={styles.layout}>
      <Toolbar />
      <div className={styles.body}>
        <div className={styles.scoreColumn}>
          <main className={styles.score}>
            <ScoreViewer />
          </main>
          <PlaybackBar />
        </div>
        <aside className={styles.panel}>
          <AIPanel />
        </aside>
      </div>
    </div>
  )
}
