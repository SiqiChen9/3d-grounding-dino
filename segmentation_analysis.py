"""
Comprehensive Segmentation Analysis Tool
Combines alignment verification, basic visualization, and detailed analysis.
Uses relative paths for GitHub collaboration.
"""
import nibabel as nib
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from PIL import Image
import argparse

# Relative paths
SEGMENTATION_PATH = "./datasets/segmentations/21057.nii"
IMAGES_DIR = "./datasets/train_images/10004/21057"
OUTPUT_DIR = "./visualizations"

os.makedirs(OUTPUT_DIR, exist_ok=True)


class SegmentationAnalyzer:
    def __init__(self):
        """Initialize analyzer with segmentation and image data."""
        self.seg_nib = nib.load(SEGMENTATION_PATH)
        self.seg_data = self.seg_nib.get_fdata()

        jpeg_files = sorted([f for f in os.listdir(IMAGES_DIR) if f.endswith('.jpeg')])
        self.image_indices = sorted([int(f.replace('.jpeg', '')) for f in jpeg_files])
        self.min_index = min(self.image_indices)
        self.max_index = max(self.image_indices)
        self.num_frames = self.seg_data.shape[2]

        self.label_colors = {
            0: [0, 0, 0, 0],        # Background
            1: [255, 0, 0, 255],    # Red
            2: [0, 255, 0, 255],    # Green
            3: [0, 0, 255, 255],    # Blue
            4: [255, 255, 0, 255],  # Yellow
            5: [255, 0, 255, 255],  # Magenta
        }

        self.study_id = "10004"  # 当前研究的ID
        self.injury_data = self._load_injury_data()

        self.label_names = {
            0: 'Background',
            1: self._get_label_with_injury('Liver'),
            2: self._get_label_with_injury('Spleen'),
            3: 'Left_Kidney',  # 左肾
            4: 'Right_Kidney',  # 右肾
            5: self._get_label_with_injury('Bowel')
        }

    def _load_injury_data(self):
        """读取器官损伤信息"""
        csv_path = "./datasets/train_2024.csv"
        try:
            df = pd.read_csv(csv_path)
            row = df[df['patient_id'] == int(self.study_id)]

            if row.empty:
                print(f"⚠ 未找到 patient_id={self.study_id} 的数据")
                return self._get_default_injury_data()

            injury_info = {}

            # 肠道：2种状态
            injury_info['bowel'] = {
                'healthy': int(row['bowel_healthy'].values[0]),
                'injury': int(row['bowel_injury'].values[0])
            }

            # 肝脏：3种等级
            injury_info['liver'] = {
                'healthy': int(row['liver_healthy'].values[0]),
                'low': int(row['liver_low'].values[0]),
                'high': int(row['liver_high'].values[0])
            }

            # 脾脏：3种等级
            injury_info['spleen'] = {
                'healthy': int(row['spleen_healthy'].values[0]),
                'low': int(row['spleen_low'].values[0]),
                'high': int(row['spleen_high'].values[0])
            }

            return injury_info

        except Exception as e:
            print(f"⚠ 无法读取损伤数据: {e}")
            return self._get_default_injury_data()

    def _get_default_injury_data(self):
        """返回默认损伤数据"""
        return {
            'bowel': {'healthy': 0, 'injury': 0},
            'liver': {'healthy': 0, 'low': 0, 'high': 0},
            'spleen': {'healthy': 0, 'low': 0, 'high': 0}
        }

    def _get_label_with_injury(self, organ_name):
        """生成包含损伤信息的标签"""
        organ_key = organ_name.lower()

        if organ_key not in self.injury_data:
            return organ_name

        states = self.injury_data[organ_key]
        active_states = [state.capitalize() for state, value in states.items() if value == 1]

        if not active_states:
            return f"{organ_name} (?)"

        status_str = "/".join(active_states)
        return f"{organ_name} ({status_str})"

    def load_image(self, image_index):
        """Load JPEG image for given index."""
        img_path = os.path.join(IMAGES_DIR, f"{image_index}.jpeg")
        if os.path.exists(img_path):
            img = np.array(Image.open(img_path))
            if len(img.shape) == 2:
                img = np.stack([img] * 3, axis=-1)
            return img
        return None

    def get_seg_frame_for_image(self, image_index):
        """Get corresponding segmentation frame."""
        position_in_range = image_index - self.min_index
        frame_index = self.num_frames - 1 - position_in_range

        if frame_index < 0 or frame_index >= self.num_frames:
            return None, None

        return frame_index, self.num_frames

    def resize_segmentation(self, seg_frame, target_size=(256, 256)):
        """Rotate and resize segmentation mask."""
        seg_frame = np.rot90(seg_frame, k=1)
        resized = cv2.resize(seg_frame, target_size, interpolation=cv2.INTER_NEAREST)
        return resized

    def verify_alignment(self):
        """Verify image-segmentation alignment."""
        print("\n" + "="*70)
        print("ALIGNMENT VERIFICATION - Study 21057")
        print("="*70)

        print(f"\n✓ Segmentation loaded")
        print(f"  Shape: {self.seg_data.shape}")
        print(f"  Data type: {self.seg_data.dtype}")

        print(f"\n✓ Images loaded")
        print(f"  Count: {len(self.image_indices)}")
        print(f"  Range: {self.min_index} to {self.max_index}")
        print(f"  Step size: {self.image_indices[1] - self.image_indices[0]}")

        print(f"\n✓ Mapping Formula Verification")
        print(f"  Total frames in segmentation: {self.num_frames}")

        # Test cases
        test_cases = [
            (self.min_index, "First image"),
            (self.max_index, "Last image"),
            ((self.min_index + self.max_index) // 2, "Middle image")
        ]

        print(f"\n✓ Alignment Test Cases")
        print(f"{'  Image':<15} {'Frame':<15} {'In Range':<12} {'Description':<20}")
        print("  " + "-"*60)

        for img_idx, desc in test_cases:
            frame_idx, _ = self.get_seg_frame_for_image(img_idx)
            in_range = "✓ Yes" if frame_idx is not None else "✗ No"
            print(f"  {img_idx:<15} {frame_idx:<15} {in_range:<12} {desc:<20}")

        print("\n" + "="*70 + "\n")

    def basic_visualization(self):
        """Generate basic overlay visualization."""
        print("BASIC VISUALIZATION")
        print("-" * 60)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        sample_indices = np.round(np.linspace(0, len(self.image_indices) - 1, 6)).astype(int)
        sample_images = [self.image_indices[i] for i in sample_indices]

        for idx, (ax, img_index) in enumerate(zip(axes.flat, sample_images)):
            image = self.load_image(img_index)
            if image is None:
                ax.axis('off')
                continue

            seg_frame_idx, _ = self.get_seg_frame_for_image(img_index)
            if seg_frame_idx is None:
                ax.axis('off')
                continue

            seg_frame = self.seg_data[:, :, seg_frame_idx]
            seg_resized = self.resize_segmentation(seg_frame)

            overlay = image.copy().astype(float)
            mask_bool = seg_resized > 0
            overlay[mask_bool] = 0.7 * overlay[mask_bool] + 0.3 * np.array([0, 255, 0])

            ax.imshow(np.clip(overlay, 0, 255).astype(np.uint8))
            ax.set_title(f'Image {img_index}', fontsize=10, fontweight='bold')
            ax.axis('off')

        plt.suptitle('Basic Segmentation Overlay Samples', fontsize=12, fontweight='bold')
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, "segmentation_basic.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved to: {output_path}\n")
        plt.close()

    def detailed_analysis(self):
        """Generate detailed analysis with statistics and multi-label overlays."""
        print("DETAILED ANALYSIS")
        print("-" * 60)

        # 1. Label statistics
        unique_labels = np.unique(self.seg_data)
        label_counts = {}
        for label in unique_labels:
            count = np.sum(self.seg_data == label)
            percentage = (count / self.seg_data.size) * 100
            label_counts[label] = (count, percentage)

        # 2. Create label distribution visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Label count bar chart
        ax = axes[0, 0]
        labels = [int(l) for l in unique_labels]
        counts = [label_counts[l][0] for l in unique_labels]
        colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
        ax.bar([str(l) for l in labels], counts, color=colors)
        ax.set_xlabel('Label', fontsize=11, fontweight='bold')
        ax.set_ylabel('Voxel Count', fontsize=11, fontweight='bold')
        ax.set_title('Label Distribution by Voxel Count', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        for i, v in enumerate(counts):
            ax.text(i, v, f'{int(v/1e6):.1f}M', ha='center', va='bottom', fontsize=9)

        # Label percentage pie chart
        ax = axes[0, 1]
        percentages = [label_counts[l][1] for l in unique_labels]
        explode = [0.05 if p > 10 else 0 for p in percentages]
        ax.pie(percentages, labels=[f'L{int(l)}' for l in labels], autopct='%1.1f%%',
               colors=colors, explode=explode, startangle=90)
        ax.set_title('Label Distribution by Percentage', fontsize=12, fontweight='bold')

        # Frame-wise presence heatmap
        ax = axes[1, 0]
        frame_presence = np.zeros((len(unique_labels), self.num_frames))
        for label_idx, label in enumerate(unique_labels):
            for frame_idx in range(self.num_frames):
                frame_presence[label_idx, frame_idx] = np.sum(self.seg_data[:, :, frame_idx] == label)

        im = ax.imshow(frame_presence, aspect='auto', cmap='hot', interpolation='nearest')
        ax.set_xlabel('Frame Index', fontsize=11, fontweight='bold')
        ax.set_ylabel('Label', fontsize=11, fontweight='bold')
        ax.set_yticks(range(len(unique_labels)))
        ax.set_yticklabels([int(l) for l in unique_labels])
        ax.set_title('Label Presence Across Frames', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Voxel Count')

        # Mapping info
        ax = axes[1, 1]
        ax.text(0.05, 0.95, "Image-Segmentation Mapping", fontsize=12, fontweight='bold',
                transform=ax.transAxes, verticalalignment='top')
        ax.text(0.05, 0.85, f"Images: {len(self.image_indices)}", fontsize=11,
                transform=ax.transAxes, verticalalignment='top')
        ax.text(0.05, 0.77, f"Range: {self.min_index}-{self.max_index}", fontsize=11,
                transform=ax.transAxes, verticalalignment='top')
        ax.text(0.05, 0.69, f"Frames: {self.num_frames}", fontsize=11,
                transform=ax.transAxes, verticalalignment='top')
        ax.text(0.05, 0.55, "Segmentation shape:", fontsize=10,
                transform=ax.transAxes, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.text(0.05, 0.40, str(self.seg_data.shape), fontsize=10,
                transform=ax.transAxes, verticalalignment='top', family='monospace')
        ax.axis('off')

        plt.suptitle('Segmentation Analysis - Study 21057', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        output_path = os.path.join(OUTPUT_DIR, "segmentation_analysis.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved to: {output_path}")
        plt.close()

        # 3. Multi-label overlays
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))

        sample_images_list = [171, 300, 500, 700, 900, 1100, 1191]
        sample_indices = np.round(np.linspace(0, len(self.image_indices) - 1, 12)).astype(int)
        sample_images_list = sorted(list(set(sample_images_list + [self.image_indices[i] for i in sample_indices])))[:12]

        # Collect valid samples
        valid_samples = []
        for img_index in sample_images_list:
            image = self.load_image(img_index)
            if image is None:
                continue

            seg_frame_idx, _ = self.get_seg_frame_for_image(img_index)
            if seg_frame_idx is None:
                continue

            valid_samples.append((img_index, image, seg_frame_idx))

        # Plot valid samples
        for idx, ax in enumerate(axes.flat):
            if idx < len(valid_samples):
                img_index, image, seg_frame_idx = valid_samples[idx]

                seg_frame = self.seg_data[:, :, seg_frame_idx]
                seg_resized = self.resize_segmentation(seg_frame)

                overlay = image.copy().astype(float)
                for label in range(1, 6):
                    mask_bool = seg_resized == label
                    color = np.array(self.label_colors[label][:3])
                    overlay[mask_bool] = 0.5 * overlay[mask_bool] + 0.5 * color

                ax.imshow(np.clip(overlay, 0, 255).astype(np.uint8))
                ax.set_title(f'Image {img_index}', fontsize=10, fontweight='bold')
                ax.axis('off')
            else:
                ax.axis('off')

        legend_elements = [
            mpatches.Patch(color=[c / 255 for c in self.label_colors[i][:3]], label=self.label_names[i])
            for i in range(1, 6)
        ]

        # 添加这一行：在图表顶部显示图例
        fig.legend(handles=legend_elements, loc='lower center', ncol=5,
                   fontsize=10, frameon=True, fancybox=True, shadow=True,
                   bbox_to_anchor=(0.5, -0.05))

        plt.suptitle('Multi-Label Segmentation Overlays', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])  # 调整布局，为图例留出空间
        output_path = os.path.join(OUTPUT_DIR, "multilabel_overlays.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved to: {output_path}\n")
        plt.close()

    def _format_injury_text(self):
        """格式化损伤信息文本"""
        sections = []

        for organ in ['liver', 'spleen', 'bowel']:
            states = self.injury_data.get(organ, {})
            active = [state.capitalize() for state, value in states.items() if value == 1]

            if active:
                status = ", ".join(active)
            else:
                status = "Unknown"

            sections.append(f"{organ.capitalize()}: {status}")

        return " | ".join(sections)

    def run_all(self):
        """Run all analyses."""
        print("\n" + "="*70)
        print("SEGMENTATION ANALYSIS TOOL")
        print("="*70)
        self.verify_alignment()
        self.basic_visualization()
        self.detailed_analysis()
        print("="*70)
        print("✓ All analyses complete!")
        print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Segmentation Analysis')
    parser.add_argument('--mode', choices=['verify', 'basic', 'detailed', 'all'],
                       default='all', help='Analysis mode to run')
    args = parser.parse_args()

    analyzer = SegmentationAnalyzer()

    if args.mode == 'verify':
        analyzer.verify_alignment()
    elif args.mode == 'basic':
        analyzer.basic_visualization()
    elif args.mode == 'detailed':
        analyzer.detailed_analysis()
    else:  # all
        analyzer.run_all()


if __name__ == "__main__":
    main()
