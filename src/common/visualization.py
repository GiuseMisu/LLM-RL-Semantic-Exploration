import matplotlib.pyplot as plt
from matplotlib import animation

def save_frames_as_gif(frames, path='./', filename='Policy.gif', fps=10, interval=100, episode_info=None):
    """
    Save a list of episode as GIF
    
    Args:
        frames: List of RGB arrays to animate
        path: Directory to save the GIF
        filename: Name of the output GIF file
        fps: Frames per second for the saved GIF (higher = faster playback)
        interval: Delay between frames in milliseconds (lower = faster animation)
        episode_info: Optional list of tuples (current_episode, total_episodes) for each frame
    """

    # figure size with extra space for title
    frame_width = frames[0].shape[1] / 72.0
    frame_height = frames[0].shape[0] / 72.0
    title_height = 0.8  # Extra height for title
    
    total_height = frame_height + title_height
    
    fig = plt.figure(figsize=(frame_width, total_height), dpi=72) #dpi = 144 for better quality
    
    # Create subplot with space for title
    # ax = plt.subplot(111)
    # ax.set_position([0, 0, 1, frame_height / (frame_height + title_height)])  # Bottom portion for image
    
    image_height_ratio = frame_height / total_height
    ax = plt.subplot(111)
    ax.set_position([0, 0, 1, image_height_ratio])

    patch = plt.imshow(frames[0])
    plt.axis('off')
    
    # # Add title in the top white space
    # title_text = plt.figtext(0.5, 0.95, '', ha='center', va='top', 
    #                         fontsize=14, fontweight='bold', color='black')
    # Position title in the white space above the image
    # Use figure coordinates where the title should be at the top
    # Place it at y position that's just above the image
    title_y = image_height_ratio + (1 - image_height_ratio) / 2
    title_text = plt.figtext(0.5, title_y, '', ha='center', va='center', 
                            fontsize=16, fontweight='bold', color='black')


    def animate(i):
        patch.set_data(frames[i])        
        # Update title with current ep
        if episode_info and i < len(episode_info):
            current_episode, total_episodes = episode_info[i]
            title_text.set_text(f"Episode {current_episode}/{total_episodes}")

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=interval)
    anim.save(path + filename, writer='pillow', fps=fps)
    plt.close()