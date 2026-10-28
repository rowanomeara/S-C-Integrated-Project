"""Generates publication-quality VIDEO_PRODUCTION_GUIDE.pdf for Lachlan using ReportLab."""

import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Adds running headers and page numbers to each page."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, total_pages):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#555555"))

        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 750, "41014 Integrated Project: Track A | 5-Minute Video Production Guide")
            self.setStrokeColor(colors.HexColor("#CCCCCC"))
            self.setLineWidth(0.5)
            self.line(54, 744, 558, 744)

        # Footer (all pages)
        footer_text = f"Page {self._pageNumber} of {total_pages}"
        self.drawRightString(558, 36, footer_text)
        self.drawString(54, 36, "Confidential: UTS Mechatronics Systems | Rowan, Jonas, Lachlan")
        self.setStrokeColor(colors.HexColor("#CCCCCC"))
        self.setLineWidth(0.5)
        self.line(54, 46, 558, 46)
        self.restoreState()


def build_pdf(output_path: str):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0f2a4a"),
        spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#334e68"),
        spaceAfter=14
    )
    h1_style = ParagraphStyle(
        "SectionH1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#0f2a4a"),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )
    h2_style = ParagraphStyle(
        "SectionH2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#102a43"),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        "BodyTextCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#243b53"),
        spaceAfter=6
    )
    callout_style = ParagraphStyle(
        "CalloutText",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#8a1c14"),
        spaceBefore=3,
        spaceAfter=3
    )
    script_meta_style = ParagraphStyle(
        "ScriptMeta",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#0b69a3"),
        spaceBefore=8,
        spaceAfter=2,
        keepWithNext=True
    )
    script_visual_style = ParagraphStyle(
        "ScriptVisual",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#486581"),
        spaceAfter=4,
        leftIndent=10
    )
    script_dialogue_style = ParagraphStyle(
        "ScriptDialogue",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13.5,
        textColor=colors.HexColor("#102a43"),
        leftIndent=14,
        rightIndent=14,
        spaceAfter=8
    )

    story = []

    # Title Banner
    story.append(Paragraph("41014 Integrated Project: Video Production Guide", title_style))
    story.append(Paragraph("Track A: TurtleBot3 Mobile Robot Visual Waypoint Navigation & Docking | Editor: Lachlan", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0f2a4a"), spaceAfter=10))

    # Critical constraints callout box
    alert_data = [[
        Paragraph(
            "CRITICAL RUBRIC RULES (KEEP STRICTLY UNDER 5:00):<br/>"
            "1. Markers stop grading at exactly 5:00. Target run time: 4:50 to 4:55.<br/>"
            "2. Physical hardware runs on both Track 1 and Track 2 are the required High Distinction evidence.<br/>"
            "3. Ground truth must only appear in simulation benchmarking comparisons. Never use it in the real-time control loop.",
            callout_style
        )
    ]]
    alert_table = Table(alert_data, colWidths=[504])
    alert_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#fce8e6")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#d9381e")),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(alert_table)
    story.append(Spacer(1, 10))

    # Section 1: Required Assets Directory
    story.append(Paragraph("1. Master Video Assets & Location Directory", h1_style))
    story.append(Paragraph("Collect these files from Google Drive and the repository before starting your editing timeline:", body_style))

    asset_data = [
        ["Asset Name", "Location", "Resolution / Format", "Role in Final 5-Minute Video"],
        ["real_mission_camera_pov.mp4", "Google Drive / Repo", "640x480 @ 20 FPS (45s)", "Onboard camera HUD with green ArUco box & telemetry"],
        ["track_01_exterior.mp4", "Google Drive", "1080p / 4K Phone Video", "Exterior wide shot of TurtleBot navigating Track 1"],
        ["track_02_exterior.mp4", "Google Drive", "1080p / 4K Phone Video", "Exterior shot of Track 2 proving altered obstacle robustness"],
        ["terminal_screen_record.mp4", "Google Drive", "1080p Screen Capture", "Live 20 Hz terminal telemetry stream with state estimates"],
        ["mujoco_sim_run.mp4", "Google Drive / Record", "1080p Screen Capture", "Simulation nominal obstacle avoidance and docking in 3D"],
        ["part_c_ekf_performance.png", "results/ in Repo", "300 DPI High-Res Image", "Proves 2-sigma EKF covariance contraction upon sighting"],
        ["part_e_robustness_trials.png", "results/ in Repo", "300 DPI High-Res Image", "10-trial Monte Carlo benchmark plot (100% success rate)"],
        ["part_e_sim_vs_real_comparison.png", "results/ in Repo", "300 DPI High-Res Image", "High Distinction quantitative sim vs real comparison"],
        ["rowan_voiceover.m4a", "Google Drive", "Clean Vocal Audio", "Speaker 1 voice track covering Segment 1 (0:00 - 1:30)"],
        ["jonas_voiceover.m4a", "Google Drive", "Clean Vocal Audio", "Speaker 2 voice track covering Segment 2 (1:30 - 3:00)"],
        ["lachlan_voiceover.m4a", "Record locally", "Clean Vocal Audio", "Speaker 3 voice track covering Segment 3 (3:00 - 4:55)"]
    ]
    t_asset = Table(asset_data, colWidths=[120, 95, 105, 184])
    t_asset.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0f2a4a")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), "Helvetica-Bold"),
        ('FONTSIZE', (0, 0), (-1, -1), 7.5),
        ('LEADING', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#d2d6dc")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_asset)
    story.append(Spacer(1, 12))

    # Section 2: Visual Layout Instructions
    story.append(Paragraph("2. Screen Layout for Hardware Demonstration (Picture-in-Picture)", h1_style))
    story.append(Paragraph(
        "During Segment 3 (physical hardware runs), do not cut back and forth between files. Use a split screen to show all angles simultaneously:",
        body_style
    ))

    layout_data = [
        ["Screen Area", "Clip Source", "Visual Details to Highlight"],
        ["Left 60% (Main View)", "Phone Exterior Recording", "Full arena view (4.0m x 2.0m). Shows TurtleBot navigating smoothly past obstacles."],
        ["Top Right 40% (Inset 1)", "real_mission_camera_pov.mp4", "Robot POV. Features green ArUco bounding box, range r, bearing phi, and [DOCKED] banner."],
        ["Bottom Right 40% (Inset 2)", "Terminal Screen Recording", "Live terminal scrolling at 20 Hz. Displays EKF estimates [x, y, theta] and LiDAR clearance."]
    ]
    t_layout = Table(layout_data, colWidths=[120, 134, 250])
    t_layout.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#334e68")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), "Helvetica-Bold"),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('LEADING', (0, 0), (-1, -1), 11),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#d2d6dc")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_layout)
    story.append(Spacer(1, 14))

    # Section 3: Simulation Recording Commands
    story.append(Paragraph("3. How to Record Fresh Simulation Footage (If Needed)", h1_style))
    story.append(Paragraph(
        "If you want to record the MuJoCo simulation window yourself, run these commands from the repository root:",
        body_style
    ))
    sim_cmd_data = [[
        Paragraph(
            "<font face='Courier' size='8'>"
            "cd S-C-Integrated-Project<br/>"
            "source .venv/bin/activate<br/>"
            "python3 main.py"
            "</font><br/><br/>"
            "Press the Space bar in the MuJoCo window to pause or unpause. Use your mouse to rotate the 3D camera. "
            "To reset the robot to a new random starting position, press the 'Reset' button in the right pane or run evaluate.py.",
            body_style
        )
    ]]
    t_sim_cmd = Table(sim_cmd_data, colWidths=[504])
    t_sim_cmd.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f0f4f8")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#bcccdc")),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(t_sim_cmd)

    story.append(PageBreak())

    # Section 4: Full Word-for-Word Narration Script
    story.append(Paragraph("4. Word-for-Word Spoken Narration Script", title_style))
    story.append(Paragraph("Divided across Rowan, Jonas, and Lachlan. Record audio in a quiet space using phone Voice Memos at ~130 words/min.", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0f2a4a"), spaceAfter=10))

    # Speaker 1: Rowan
    story.append(Paragraph("SPEAKER 1: ROWAN (0:00 - 1:30 | 90 Seconds | 185 Words)", script_meta_style))
    story.append(Paragraph("Visuals: Architecture diagram, state equations, coordinate frame transformation chain, split MuJoCo and real arena view.", script_visual_style))
    story.append(Paragraph(
        "\"Welcome to our 41014 Integrated Project demonstration for Track A: Autonomous Mobile Robot Navigation and Visual Docking. "
        "Our team consists of Jonas, Lachlan, and Rowan. Our platform is a TurtleBot3 operating under unicycle kinematics, governed by state vector x, y, and theta, with forward velocity v and angular velocity omega as control inputs.<br/><br/>"
        "In our model analysis script, we evaluate the Lie algebra rank condition using the platform vector fields and their Lie bracket. "
        "The distribution determinant equals one across all orientations, proving small-time local controllability. "
        "For sensing, we map monocular camera pixels through intrinsic parameters into the camera frame, transform to the robot body frame, and project into the world frame.<br/><br/>"
        "We also evaluate the system observability matrix. Global pose is unobservable from wheel odometry alone due to dead-reckoning drift. "
        "Once the monocular camera detects the ArUco marker, the measurement Jacobian restores full rank-three observability. "
        "Crucially, our controller acts strictly on estimated states from the filter and never reads ground truth coordinates.\"",
        script_dialogue_style
    ))
    story.append(Spacer(1, 4))

    # Speaker 2: Jonas
    story.append(Paragraph("SPEAKER 2: JONAS (1:30 - 3:00 | 90 Seconds | 195 Words)", script_meta_style))
    story.append(Paragraph("Visuals: MuJoCo 3D simulation running, results/part_c_ekf_performance.png, results/part_e_robustness_trials.png.", script_visual_style))
    story.append(Paragraph(
        "\"Here is our baseline simulation running in MuJoCo. The robot starts from a randomized pose, scans the environment with a 2D LiDAR, and detects the central obstacle. "
        "To estimate state under sensor noise, we implemented an Extended Kalman Filter fusing wheel encoder odometry with camera range and bearing observations.<br/><br/>"
        "We tuned the process noise covariance matrix Q to reflect unmodelled wheel slip on floor surfaces over our 50-millisecond sample time, and measurement noise matrix R to match camera corner quantization. "
        "As seen in our EKF performance plot, position uncertainty bounds widen steadily while the robot bypasses the obstacle under dead reckoning.<br/><br/>"
        "At ten seconds, the camera sights ArUco Marker 0. The innovation covariance yields an immediate contraction in the filter covariance, pulling position error down to under three centimetres. "
        "Across our ten-trial Monte Carlo robustness benchmark with randomized starting locations, our state-feedback pipeline achieved a one hundred percent docking rate, with a mean final position error of only 3.8 centimetres and heading alignment within 1.2 degrees.\"",
        script_dialogue_style
    ))
    story.append(Spacer(1, 4))

    # Speaker 3: Lachlan
    story.append(Paragraph("SPEAKER 3: LACHLAN (3:00 - 4:55 | 115 Seconds | 245 Words)", script_meta_style))
    story.append(Paragraph("Visuals: Split screen Track 1 and Track 2 exterior + real_mission_camera_pov.mp4 HUD + results/part_e_sim_vs_real_comparison.png.", script_visual_style))
    story.append(Paragraph(
        "\"To satisfy the High Distinction requirements, we deployed our complete software pipeline onto the physical TurtleBot3 in the UTS robotics laboratory using rosbridge WebSocket. "
        "Here is Track 1 on the 4.0-metre by 2.0-metre arena. On the right, the camera HUD displays our live ArUco detection box, distance r, and bearing phi.<br/><br/>"
        "Our LiDAR processor monitors a 100-degree frontal cone. When obstacles approach within 55 centimetres, the robot executes a reactive bypass. "
        "If an obstacle enters within 32 centimetres, the controller commands an emergency zero-velocity brake, rotating in place until the path clears. "
        "Once the marker locks, the controller transitions to Position-Based Visual Servoing and terminal corridor line-tracking, docking within three centimetres of the plate. "
        "Here is Track 2 with an altered starting condition and obstacle layout, confirming reliable docking.<br/><br/>"
        "In our quantitative comparison plot, Panel 1 shows that physical hardware trajectories match simulation topology closely. "
        "Panel 2 confirms that real-world 2-sigma uncertainty contracts upon visual acquisition identically to simulation.<br/><br/>"
        "During hardware deployment, we diagnosed two real-world issues. First, Wi-Fi transport latency caused steering oscillations under our initial visual servoing gain. "
        "We damped the gain to 0.65 and added an acceleration slew limiter to eliminate hunting. "
        "Second, we resolved the close-range waypoint singularity using terminal corridor tracking. "
        "Our pipeline proves robust, observable, and fully closed-loop docking across both simulation and real hardware. Thank you.\"",
        script_dialogue_style
    ))

    story.append(PageBreak())

    # Section 5: Pre-Export QC Checklist
    story.append(Paragraph("5. Final Video Quality Control & Submission Checklist", h1_style))
    story.append(Paragraph("Run through this checklist before uploading the MP4 to YouTube and Canvas:", body_style))

    qc_data = [
        ["Check Item", "Target Standard", "Why It Affects Your Grade"],
        ["Total Duration", "Strictly 4:50 to 4:55 (Under 5:00)", "Rubric Rule: Content past 5:00 is ignored. If video is 5:05, you lose the conclusion."],
        ["Audio Balancing", "Normalize all 3 voices to -14 LUFS", "Rowan, Jonas, and Lachlan must have matched volume without background hum or clipping."],
        ["Resolution", "1080p (1920x1080) @ 30 or 60 FPS", "Ensures text overlays, terminal output, and plot axes remain sharp and readable."],
        ["Ground Truth Label", "Visible ONLY in Sim Benchmarks", "Rubric Rule: Using ground truth in real-time control caps grade at Pass (0-49)."],
        ["YouTube Visibility", "Set to 'Unlisted' (NOT Private)", "If set to Private, markers cannot open the link and will assign zero for video."],
        ["Title Card Info", "Group Number, Member Names & IDs", "Must appear in the first 5 seconds or final 5 seconds to match Canvas submission."]
    ]
    t_qc = Table(qc_data, colWidths=[110, 150, 244])
    t_qc.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0f2a4a")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), "Helvetica-Bold"),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('LEADING', (0, 0), (-1, -1), 11),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#d2d6dc")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t_qc)
    story.append(Spacer(1, 16))

    # Project metadata summary
    meta_p = Paragraph(
        "<b>Submission Metadata:</b><br/>"
        "• Subject: 41014 Sensors and Control in Mechatronic Systems (UTS)<br/>"
        "• Project: Track A Mobile Robot Visual Waypoint Navigation & Docking<br/>"
        "• Team: Rowan O'Meara, Jonas, Lachlan<br/>"
        "• Generated PDF Guide: VIDEO_PRODUCTION_GUIDE.pdf",
        body_style
    )
    story.append(meta_p)

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated: {output_path}")


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    out_pdf = os.path.join(out_dir, "VIDEO_PRODUCTION_GUIDE.pdf")
    build_pdf(out_pdf)
