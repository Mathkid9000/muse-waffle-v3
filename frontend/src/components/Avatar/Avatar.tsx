import type { Profile } from "../../types";
import styles from "../Avatar/Avatar.module.css"

interface AvatarProps {
    profile: Profile | null
}

export default function Avatar({ profile }: AvatarProps) {
    return  (
        <div className={styles.avatarSquare}>
            {profile && profile.avatar_url ? (
                <img src={profile.avatar_url} className={styles.avatarImage} />
            ) : (profile ? (
                <p className={styles.avatarLetter}>{profile.username![0].toUpperCase()}</p>
            ) : (
                <p className={styles.avatarLetter}>?</p>
            ))}
        </div>
    )
}