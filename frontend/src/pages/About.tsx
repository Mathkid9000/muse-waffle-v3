import { Link } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import styles from './About.module.css'
import Screenshot1 from '../assets/about/Screenshot1.png'
import Screenshot2 from '../assets/about/Screenshot2.png'
import Figure2 from '../assets/about/Figure2.png'

function Figure({ image, label, small }: { image?: string; label: string; small?: boolean }) {
  return image ? (
    <img className={styles.figure} src={image} alt={label} style={(small ?? false) ? { width: "50%", margin: '30px auto' }: {}} />
  ) : (
    <div className={`${styles.figure} ${styles.figurePlaceholder}`} role="img" aria-label={label}>
      {label}
    </div>
  )
}

export default function About() {
  const { user } = useAuth()

  return (
    <>
      <header className={styles.headerBar}>
        <Link to={user ? '/' : '/login'} className={styles.titleLink}>
          <h1 className={styles.museWaffleTitle}>MUSEWAFFLE</h1>
        </Link>
        <Link to={user ? '/' : '/login'} className={styles.headerButton}>{user ? 'Dashboard' : 'Sign in'}</Link>
      </header>
      <div className={styles.background}></div>
      <main className={styles.page}>
        <section className={styles.hero}>
          <div className={`${styles.heroTitle} ${styles.waveText}`}>
            <span>M</span>
            <span>U</span>
            <span>S</span>
            <span>E</span>
            <span>&nbsp;</span>
            <wbr></wbr>
            <span>W</span>
            <span>A</span>
            <span>F</span>
            <span>F</span>
            <span>L</span>
            <span>E</span>
          </div>
          <p className={styles.heroText}>
            My attempt at symbolic music generation.
          </p>
          <Figure image={Screenshot1} label="Hero image" />
        </section>

        <section className={styles.hero}>
          <h2 className={styles.sectionTitle}>Structural Overview</h2>
          <p className={styles.bodyText}>
            Muse Waffle attempts to blend creative tasks like melodic generation with deterministic tasks like adding instruments.
            When the user enters a prompt, it makes a call to the <strong style={{ textDecoration: 'underline' }}>Claude API</strong> that determines whether or not the creative model is needed. However, this creates an information problem. 
            <br></br><br></br><em className={styles.heroText}>How do we get the LLM to tell the creative music model what to generate?</em>
          </p>
          <div style={{ padding: '20px' }}></div>

          <div className={styles.step}>
            <div className={styles.stepText}>
              <h3 className={styles.stepTitle}>Scores vs. Templates</h3>
              <p className={styles.bodyText}>
                My solution is to allow the LLM to mask elements of sheet music (ex. rhythm, pitches, chords, etc...) through the creation of templates.
                By obscuring parts of a score, the creative model can generate new content while staying within the bounds of the user's prompt.
              </p>
            </div>
            <Figure image={Screenshot2} label={'Templates vs Scores'} />
          </div>
          <p className={styles.bodyText}> 
            This strategy really shines when writing parts that follow form. Currently, symbolic music generation is known for its temperamental and often sporadic passages, due to the naturally long token context of multitrack music.
            This is sidestepped by using both templates and a token-efficient architecture.
          </p>
        </section>

        <section className={styles.hero}>
          <h2 className={styles.sectionTitle}>Model Architecture</h2>
          <p className={styles.bodyText}> 
            To create the music model, I implement the token structure outlined in the Zhu et al. paper, "<a href="https://arxiv.org/pdf/2409.00919" style={{ textDecoration: 'underline', color: 'inherit', fontWeight: '700' }}>MMT-BERT: Chord-aware Symbolic Music Generation Based on Multitrack Music Transformer and MusicBERT</a>."
            In this paper, it outlines a quintuple token where each represents a single musical element (note, chord, instrument, clef, ...) containing additional details (ex. pitch, measure number, duration). As a bonus, the model is trained with chords in its dataset, giving it the ability to write music around prompted harmony (returning to the idea of templates). 
          </p>
          <Figure image={Figure2} label={'Quintuple tokens'} small={true}/>
          <p className={styles.bodyText} style={{ fontStyle: 'italic', marginTop: '-20px' }}>A table extracted from the paper illustrating the quintuple token in practice.</p>
        </section>

        <section className={styles.hero}>
          <h2 className={styles.sectionTitle}>Training and Dataset</h2>
          <p className={styles.bodyText}>
            Muse Waffle uses a transformer architecture, not the GAN structure presented in the paper. While this was tested, I found less success as the GAN frequently collapsed in on itself. The current transformer is trained on data from&nbsp;
            <a href="https://fourscoreandmore.org/openscore/" style={{ textDecoration: 'underline', color: 'inherit', fontWeight: '700' }}>OpenScore</a> and&nbsp;
            <a href="https://github.com/pnlong/PDMX/" style={{ textDecoration: 'underline', color: 'inherit', fontWeight: '700' }}>PDMX</a>.
          </p>
        </section>

        <section className={styles.cta}>
          <h2 className={styles.sectionTitle}>Try it!</h2>
          <p className={styles.bodyText}>Note: requires a Claude API Key</p>
          <Link to={user ? '/' : '/login'} className={styles.ctaButton}>{user ? 'Open Dashboard' : 'Get Started'}</Link>
        </section>
      </main>
    </>
  )
}
