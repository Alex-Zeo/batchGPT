# flake8: noqa
# mypy: ignore-errors
import streamlit as st
import asyncio
import pandas as pd
import plotly.express as px
import time
import os
import glob
from datetime import datetime
from app.openai_batch import (
    run_batch,
    retrieve_batch_status,
    retrieve_batch_results,
    list_available_models,
    list_batch_compatible_models,
    refresh_api_key,
)
from utils.validators import is_valid_poll_interval, validate_api_key
from utils.formatters import (
    format_batch_results,
    calculate_cost_estimate,
)
from app.file_processor import (
    process_uploaded_file,
    process_multiple_files,
    split_text_into_chunks,
    detect_file_type,
    generate_summary,
)
from app.batch_manager import batch_manager
from dotenv import load_dotenv
from app.logger import logger, setup_logger

# Set up logging
setup_logger()

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="OpenAI o1-Pro Batch API Interface",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Simple card component using standard Streamlit
def discord_card(title, content, key=None):
    with st.container():
        st.subheader(title)
        st.markdown(content, unsafe_allow_html=True)


# Application title and description
st.title("🚀 OpenAI o1-Pro Batch API Interface")
st.markdown(
    "This application allows you to submit batch requests to OpenAI's API with advanced controls and monitoring features."
)

# Initialize session state for storing data between reruns
if "available_models" not in st.session_state:
    st.session_state.available_models = []
if "batch_jobs" not in st.session_state:
    st.session_state.batch_jobs = {}
if "polling_active" not in st.session_state:
    st.session_state.polling_active = False
if "start_time" not in st.session_state:
    st.session_state.start_time = time.time()
if "api_key_valid" not in st.session_state:
    st.session_state.api_key_valid = validate_api_key()
if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = ""
if "file_type" not in st.session_state:
    st.session_state.file_type = ""
if "show_logs" not in st.session_state:
    st.session_state.show_logs = False
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "text_input" not in st.session_state:
    st.session_state.text_input = ""


# Function to get available models
async def get_models():
    # Get all models and batch-compatible models
    all_models = await list_available_models()
    batch_models = await list_batch_compatible_models()

    # Store both in session state
    st.session_state.available_models = all_models
    st.session_state.batch_compatible_models = batch_models

    return batch_models  # Return only batch-compatible models for selection


# Function to read recent logs
def get_recent_logs(log_type="info", max_lines=100):
    log_path = os.path.join("logs", f"{log_type}.log")
    if not os.path.exists(log_path):
        return ["No log file found"]

    try:
        with open(log_path, "r") as log_file:
            # Read last N lines
            lines = log_file.readlines()
            return lines[-max_lines:] if len(lines) > max_lines else lines
    except Exception as e:
        return [f"Error reading logs: {str(e)}"]


# Sidebar for configuration settings
with st.sidebar:
    st.header("API Configuration")

    # API Key input (masked)
    api_key = st.text_input(
        "OpenAI API Key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
        help="Enter your OpenAI API key. It will be stored in the .env file.",
    )

    if api_key:
        # Update API key in environment and .env file if changed
        if api_key != os.getenv("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = api_key
            with open(".env", "w") as f:
                f.write(f"OPENAI_API_KEY={api_key}")
            # Update the OpenAI client configuration
            refresh_api_key()
            # Validate the key
            st.session_state.api_key_valid = validate_api_key()

    # Display API key status
    if not api_key:
        st.warning("⚠️ No API key provided. Please enter your OpenAI API key.")
    elif not st.session_state.api_key_valid:
        st.error("❌ Invalid API key. Please check your OpenAI API key.")
    else:
        st.success("✅ API key validated")

    # Get available models button
    if st.button("Refresh Available Models"):
        if not st.session_state.api_key_valid:
            st.error("Cannot fetch models: Invalid API key")
        else:
            with st.spinner("Fetching available models..."):
                try:
                    models = asyncio.run(get_models())
                    if models:
                        st.success(f"Found {len(models)} models")
                    else:
                        st.warning(
                            "No models found or API key may not have access to models"
                        )
                except Exception as e:
                    st.error(f"Error fetching models: {str(e)}")

    st.header("Batch Settings")

    # Model selection
    if not st.session_state.available_models and st.session_state.api_key_valid:
        # Run once to populate models
        with st.spinner("Fetching available models..."):
            try:
                models = asyncio.run(get_models())
            except Exception as e:
                st.error(f"Could not fetch models: {str(e)}")

    # Default to o1-pro if available, otherwise use the first model
    default_model = (
        "o1-pro"
        if "o1-pro" in st.session_state.available_models
        else (
            st.session_state.available_models[0]
            if st.session_state.available_models
            else "o1-pro"
        )
    )

    model = st.selectbox(
        "Model",
        options=(
            st.session_state.available_models
            if st.session_state.available_models
            else ["o1-pro"]
        ),
        index=(
            st.session_state.available_models.index(default_model)
            if default_model in st.session_state.available_models
            else 0
        ),
        help="Select the model to use for this batch request",
    )

    # Advanced model settings
    reasoning_effort = st.selectbox(
        "Reasoning Effort",
        options=["low", "medium", "high"],
        index=1,
        help="Control how much internal reasoning the model does (o1 models only)",
    )

    # Use text inputs for numeric parameters as required
    temperature_str = st.text_input(
        "Temperature",
        value="0.7",
        help="Controls randomness. Lower values are more deterministic.",
    )

    max_tokens_str = st.text_input(
        "Max Tokens",
        value="1024",
        help="Maximum number of tokens to generate in the response.",
    )

    max_batch_size_str = st.text_input(
        "Max Batch Size",
        value="5000",
        help="Maximum number of requests in a single batch. Larger batches will be split.",
    )

    budget_str = st.text_input(
        "Budget (USD)",
        value="",
        help="Abort submission if estimated cost would exceed this amount. Leave blank for no limit.",
    )

    # Response format
    response_format = st.selectbox(
        "Response Format",
        options=["json_object", "text"],
        index=0,
        help="Format of the model's response",
    )

    # Polling settings
    poll_interval_str = st.text_input(
        "Poll Interval",
        value="30s",
        help="Interval to check batch status (e.g., 30s, 5m, 1h)",
    )

    # File processing settings
    st.header("File Processing Settings")

    extract_tables = st.checkbox(
        "Extract Tables from PDFs/DOCs",
        value=False,
        help="Extract tables from PDF and DOCX files and include them in text",
    )

    file_process_mode = st.selectbox(
        "Multiple File Handling",
        options=["separate", "combine", "zip"],
        index=0,
        help="How to handle multiple files: as separate prompts, combined into one, or extract files from ZIP",
    )

    # Chunk size for file processing
    max_chunk_size_str = st.text_input(
        "Max Chunk Size",
        value="4000",
        help="Maximum size of text chunks for processing large files",
    )

    chunk_overlap_str = st.text_input(
        "Chunk Overlap",
        value="200",
        help="Overlap between text chunks to maintain context",
    )

    # Show logs toggle
    st.session_state.show_logs = st.checkbox(
        "Show Logs", value=st.session_state.show_logs
    )

    # Convert string inputs to appropriate types with validation
    try:
        temperature = float(temperature_str)
        if not (0 <= temperature <= 2):
            st.sidebar.warning(
                "Temperature should be between 0 and 2. Using default of 0.7."
            )
            temperature = 0.7
    except ValueError:
        st.sidebar.warning("Invalid temperature value. Using default of 0.7.")
        temperature = 0.7

    try:
        max_tokens = int(max_tokens_str)
        if max_tokens <= 0:
            st.sidebar.warning(
                "Max tokens should be a positive integer. Using default of 1024."
            )
            max_tokens = 1024
    except ValueError:
        st.sidebar.warning("Invalid max tokens value. Using default of 1024.")
        max_tokens = 1024

    try:
        max_batch_size = int(max_batch_size_str)
        if not (100 <= max_batch_size <= 10000):
            st.sidebar.warning(
                "Batch size should be between 100 and 10000. Using default of 5000."
            )
            max_batch_size = 5000
    except ValueError:
        st.sidebar.warning("Invalid batch size value. Using default of 5000.")
        max_batch_size = 5000

    try:
        budget = float(budget_str) if budget_str else None
        if budget is not None and budget <= 0:
            st.sidebar.warning("Budget should be positive. Ignoring budget limit.")
            budget = None
    except ValueError:
        st.sidebar.warning("Invalid budget value. Ignoring budget limit.")
        budget = None

    try:
        max_chunk_size = int(max_chunk_size_str)
        if max_chunk_size < 100:
            st.sidebar.warning(
                "Chunk size should be at least 100. Using default of 4000."
            )
            max_chunk_size = 4000
    except ValueError:
        st.sidebar.warning("Invalid chunk size value. Using default of 4000.")
        max_chunk_size = 4000

    try:
        chunk_overlap = int(chunk_overlap_str)
        if chunk_overlap < 0:
            st.sidebar.warning(
                "Chunk overlap should be non-negative. Using default of 200."
            )
            chunk_overlap = 200
    except ValueError:
        st.sidebar.warning("Invalid chunk overlap value. Using default of 200.")
        chunk_overlap = 200

    poll_interval_seconds = is_valid_poll_interval(poll_interval_str)
    if poll_interval_seconds is None:
        st.sidebar.warning("Invalid poll interval. Using default of 30s.")
        poll_interval_seconds = 30

# Main interface with tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Submit Batch", "Monitor Batches", "Results", "Statistics", "Logs"]
)

# Tab 1: Submit Batch
with tab1:
    st.header("Submit Batch Requests")

    # Show API key warning if needed
    if not st.session_state.api_key_valid:
        st.error(
            "⚠️ Please enter a valid OpenAI API key in the sidebar before submitting batches."
        )

    # Add file upload option for multiple files
    st.subheader("Upload Files (Optional)")

    uploaded_files = st.file_uploader(
        "Upload files to extract text for batch processing",
        type=["txt", "py", "pdf", "docx", "zip", "js", "html", "css", "json"],
        accept_multiple_files=True,
        help="Upload multiple files to process as separate prompts or combine them",
    )

    if uploaded_files:
        process_button = st.button("Process Files")

        if process_button:
            with st.spinner(f"Processing {len(uploaded_files)} files..."):
                # Process multiple files
                processed_files = process_multiple_files(
                    uploaded_files,
                    extract_tables=extract_tables,
                    process_mode=file_process_mode,
                )
                st.session_state.processed_files = processed_files

                if processed_files:
                    # Generate a summary
                    summary = generate_summary(processed_files)
                    st.success(f"Successfully processed {len(processed_files)} files")

                    # Display summary in a Discord-style card
                    discord_card("File Processing Summary", summary)

                    # Ask if user wants to convert processed files to prompts
                    use_as_prompts = st.checkbox(
                        "Use processed files as prompts", value=True
                    )

                    if use_as_prompts:
                        # Prepare text for prompts
                        if file_process_mode == "combine" or len(processed_files) == 1:
                            prompts = [processed_files[0]["text"]]
                        else:
                            prompts = []
                            for file in processed_files:
                                # Check if text is too long and needs chunking
                                if len(file["text"]) > max_chunk_size:
                                    chunks = split_text_into_chunks(
                                        file["text"], max_chunk_size, chunk_overlap
                                    )
                                    for i, chunk in enumerate(chunks):
                                        prompts.append(
                                            f"[{file['filename']} - Part {i+1}/{len(chunks)}] {chunk}"
                                        )
                                else:
                                    prompts.append(
                                        f"[{file['filename']}] {file['text']}"
                                    )

                        # Set prompts in the text area
                        st.session_state.text_input = "\n---\n".join(prompts)

    # Display each processed file in an expander if available
    if st.session_state.processed_files:
        st.subheader("Processed Files")
        file_tabs = st.tabs(
            [file["filename"] for file in st.session_state.processed_files[:10]]
        )

        for i, tab in enumerate(file_tabs):
            with tab:
                file = st.session_state.processed_files[i]

                # Display metadata as a formatted table
                metadata_html = (
                    "<table style='width: 100%; border-collapse: collapse;'>"
                )
                for key, value in file["metadata"].items():
                    if key not in ["filename", "file_size", "processed_at"]:
                        metadata_html += f"<tr><td style='padding: 4px; color: #B9BBBE;'>{key}</td><td style='padding: 4px;'>{value}</td></tr>"
                metadata_html += "</table>"

                discord_card("File Metadata", metadata_html)

                # Preview text content
                if len(file["text"]) > 5000:
                    preview = file["text"][:5000] + "... (truncated)"
                else:
                    preview = file["text"]

                st.code(preview, language="text")

    # Multi-line text input for prompts with default value if set
    user_inputs = st.text_area(
        "Enter your prompts (separate with '---')",
        value=st.session_state.text_input,
        height=300,
        placeholder="What is the capital of France?\n---\nExplain quantum computing in simple terms\n---\nWrite a short poem about artificial intelligence",
    )

    # Submit button with batch creation logic
    submit_disabled = not st.session_state.api_key_valid
    if submit_disabled:
        st.info("Please enter a valid API key to enable batch submission")

    if st.button("Submit Batch", disabled=submit_disabled):
        if not user_inputs.strip():
            st.error("Please enter at least one prompt!")
        else:
            # Parse prompts
            prompts = [p.strip() for p in user_inputs.strip().split("---") if p.strip()]

            if len(prompts) == 0:
                st.error("No valid prompts provided!")
            else:
                # Show batch summary before submission
                batch_preview = {
                    "model": model,
                    "prompts_count": len(prompts),
                    "reasoning_effort": reasoning_effort,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }

                progress_bar = st.progress(0)
                status_text = st.empty()

                # Submit batch with progress updates
                with st.spinner(f"Submitting batch of {len(prompts)} prompts..."):
                    try:
                        status_text.info("Preparing batch request...")
                        progress_bar.progress(10)

                        # Ensure API key is refreshed
                        refresh_api_key()
                        progress_bar.progress(20)

                        # Run batch with all the parameters
                        status_text.info("Submitting to OpenAI batch API...")
                        batch_job = asyncio.run(
                            run_batch(
                                inputs=prompts,
                                model=model,
                                reasoning_effort=reasoning_effort,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                response_format=response_format,
                                max_batch_size=max_batch_size,
                                budget=budget,
                            )
                        )

                        progress_bar.progress(90)
                        status_text.info("Finalizing submission...")

                        # Handle single batch or multiple batches
                        if isinstance(batch_job, list):
                            progress_bar.progress(100)
                            status_text.success(
                                f"Submitted {len(batch_job)} batches successfully!"
                            )

                            # Create result card with all batch IDs
                            batch_ids = [job.id for job in batch_job]
                            result_html = f"<p>Submitted {len(batch_job)} batches with {len(prompts)} total prompts.</p>"
                            result_html += "<p>Batch IDs:</p><ul>"

                            for i, batch_id in enumerate(batch_ids):
                                result_html += f"<li><code>{batch_id}</code></li>"

                            result_html += "</ul>"
                            discord_card("Batch Submission Results", result_html)
                        else:
                            progress_bar.progress(100)
                            status_text.success(f"Batch submitted successfully!")

                            # Create result card
                            result_html = f"""
                            <p>Batch ID: <code>{batch_job.id}</code></p>
                            <p>Status: <span style="color: #FAA61A;">{batch_job.status}</span></p>
                            <p>Created At: {batch_job.created_at}</p>
                            <p>Prompts: {len(prompts)}</p>
                            """
                            discord_card("Batch Submission Results", result_html)
                    except ValueError as e:
                        # Handle API key validation errors
                        if "API key" in str(e):
                            st.error(f"API Key Error: {str(e)}")
                            st.info("Please update your API key in the sidebar.")
                        else:
                            st.error(f"Validation Error: {str(e)}")
                    except Exception as e:
                        st.error(f"Error submitting batch: {str(e)}")
                        if (
                            "401" in str(e)
                            or "authentication" in str(e).lower()
                            or "api key" in str(e).lower()
                        ):
                            st.info(
                                "This appears to be an authentication issue. Please check your API key."
                            )

# Tab 2: Monitor Batches
with tab2:
    st.header("Monitor Batch Status")

    # Show API key warning if needed
    if not st.session_state.api_key_valid:
        st.error(
            "⚠️ Please enter a valid OpenAI API key in the sidebar to monitor batches."
        )

    # Refresh the batch list from batch manager
    st.session_state.batch_jobs = {
        batch_id: batch_manager.get_batch(batch_id)
        for batch_id in batch_manager.get_all_batch_ids()
    }

    # Batch scan and cleanup
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("Scan for New Batches"):
            with st.spinner("Scanning batch folder..."):
                batch_manager.scan_batch_folder()
                # Update session state with the latest batches
                st.session_state.batch_jobs = {
                    batch_id: batch_manager.get_batch(batch_id)
                    for batch_id in batch_manager.get_all_batch_ids()
                }
                st.success(f"Found {len(st.session_state.batch_jobs)} batches")

    with col2:
        # Add auto-polling functionality
        if "polling_active" not in st.session_state:
            st.session_state.polling_active = False

        if st.session_state.polling_active:
            poll_button_label = "Stop Polling"
        else:
            poll_button_label = "Start Auto-Polling"

        if st.button(poll_button_label):
            st.session_state.polling_active = not st.session_state.polling_active

        # Display polling status
        if st.session_state.polling_active:
            st.info(
                "🔄 Auto-polling is active. Checking batch status every "
                + f"{poll_interval_seconds} seconds."
            )

    with col3:
        # Allow refreshing a specific batch
        in_progress_batches = [
            bid
            for bid, batch in st.session_state.batch_jobs.items()
            if batch and batch.get("status") in ["in_progress", "validating"]
        ]

        if in_progress_batches:
            selected_batch = st.selectbox(
                "Select batch to refresh",
                options=in_progress_batches,
                key="refresh_batch_select",
            )

            if st.button("Refresh Selected Batch"):
                with st.spinner(f"Refreshing status of batch {selected_batch}..."):
                    try:
                        # Ensure API key is refreshed
                        refresh_api_key()

                        # Get updated status
                        batch_status = asyncio.run(
                            retrieve_batch_status(selected_batch)
                        )

                        # Get results if completed
                        if batch_status.status == "completed":
                            results, errors = asyncio.run(
                                retrieve_batch_results(batch_status)
                            )

                            # Update batch manager
                            batch_manager.update_batch(
                                selected_batch,
                                status=batch_status.status,
                                results=results,
                                errors=errors,
                            )

                            st.success(
                                f"Batch {selected_batch} is completed with {len(results)} results and {len(errors)} errors"
                            )
                        else:
                            # Update status only
                            batch_manager.update_batch(
                                selected_batch, status=batch_status.status
                            )

                            st.info(
                                f"Batch {selected_batch} status: {batch_status.status}"
                            )

                        # Update session state
                        st.session_state.batch_jobs[selected_batch] = (
                            batch_manager.get_batch(selected_batch)
                        )
                    except Exception as e:
                        st.error(f"Error refreshing batch: {str(e)}")
        else:
            st.info("No in-progress batches to refresh")

    # Option to manually add a batch ID
    st.subheader("Add Batch by ID")
    col1, col2 = st.columns([3, 1])
    with col1:
        batch_status_id = st.text_input(
            "Add Batch ID", placeholder="Enter a batch ID to monitor"
        )
    with col2:
        add_batch_disabled = not st.session_state.api_key_valid
        if (
            st.button("Add Batch", disabled=add_batch_disabled)
            and batch_status_id.strip()
        ):
            if batch_status_id in st.session_state.batch_jobs:
                st.warning(f"Batch {batch_status_id} is already being monitored.")
            else:
                with st.spinner(f"Getting info for batch {batch_status_id}..."):
                    try:
                        # Ensure API key is refreshed
                        refresh_api_key()

                        batch_status = asyncio.run(
                            retrieve_batch_status(batch_status_id.strip())
                        )
                        batch_info = {
                            "id": batch_status_id,
                            "status": batch_status.status,
                            "created_at": batch_status.created_at,
                            "model": (
                                batch_status.model
                                if hasattr(batch_status, "model")
                                else "Unknown"
                            ),
                            "request_count": "Unknown",
                            "results_count": 0,
                            "errors_count": 0,
                        }

                        # Add to batch manager
                        batch_manager.add_batch(batch_status_id, batch_info)

                        # Update session state
                        st.session_state.batch_jobs[batch_status_id] = batch_info

                        st.success(f"Added batch {batch_status_id} for monitoring.")
                    except ValueError as e:
                        # Handle API key validation errors
                        if "API key" in str(e):
                            st.error(f"API Key Error: {str(e)}")
                            st.info("Please update your API key in the sidebar.")
                        else:
                            st.error(f"Validation Error: {str(e)}")
                    except Exception as e:
                        st.error(f"Error retrieving batch: {str(e)}")
                        if "404" in str(e) or "not found" in str(e).lower():
                            st.warning(
                                f"Batch ID '{batch_status_id}' not found. Please check the ID."
                            )
                        elif (
                            "401" in str(e)
                            or "authentication" in str(e).lower()
                            or "api key" in str(e).lower()
                        ):
                            st.info(
                                "This appears to be an authentication issue. Please check your API key."
                            )

    # List all tracked batches with their status
    if st.session_state.batch_jobs:
        st.subheader("Tracked Batches")

        # Create filter options
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            status_filter = st.multiselect(
                "Filter by Status",
                options=["in_progress", "validating", "completed", "failed", "unknown"],
                default=[],
            )

        with filter_col2:
            model_filter = st.multiselect(
                "Filter by Model",
                options=set(
                    batch.get("model", "Unknown")
                    for batch in st.session_state.batch_jobs.values()
                    if batch
                ),
                default=[],
            )

        # Create a dataframe for batches with filtering
        batch_data = []
        for batch_id, batch_info in st.session_state.batch_jobs.items():
            if batch_info:  # Only include valid batches
                # Apply filters
                if status_filter and batch_info.get("status") not in status_filter:
                    continue
                if model_filter and batch_info.get("model") not in model_filter:
                    continue

                batch_data.append(
                    {
                        "Batch ID": batch_id,
                        "Status": batch_info.get("status", "Unknown"),
                        "Created At": batch_info.get("created_at", "Unknown"),
                        "Model": batch_info.get("model", "Unknown"),
                        "Requests": batch_info.get("request_count", "Unknown"),
                        "Results": len(batch_info.get("results", [])),
                        "Errors": len(batch_info.get("errors", [])),
                    }
                )

        # Custom styling for status colors in the dataframe
        def color_status(val):
            color = "#DCDDDE"  # Default
            if val == "completed":
                color = "#43B581"  # Green
            elif val == "failed" or val == "expired" or val == "cancelled":
                color = "#F04747"  # Red
            elif val == "in_progress" or val == "validating":
                color = "#FAA61A"  # Yellow
            return f"color: {color}"

        batch_df = pd.DataFrame(batch_data)
        if not batch_df.empty:
            st.dataframe(
                batch_df.style.applymap(color_status, subset=["Status"]),
                use_container_width=True,
            )

            # Add option to download batch list
            csv = batch_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Batch List as CSV",
                csv,
                f"batch_list_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv",
                key="download-batch-list",
            )
        else:
            st.info("No batches match the selected filters.")

    # Add background polling using Streamlit's rerun mechanism
    if st.session_state.polling_active and st.session_state.api_key_valid:
        # Get all in-progress batches
        in_progress_batch_ids = [
            bid
            for bid, batch in st.session_state.batch_jobs.items()
            if batch and batch.get("status") in ["in_progress", "validating"]
        ]

        if in_progress_batch_ids:
            # Select one batch to update per rerun to avoid rate limits
            batch_to_update = in_progress_batch_ids[0]

            # Update the batch status
            try:
                logger.info(f"Polling batch {batch_to_update}")
                refresh_api_key()
                batch_status = asyncio.run(retrieve_batch_status(batch_to_update))

                # If status changed to completed, get results
                if batch_status.status == "completed":
                    results, errors = asyncio.run(retrieve_batch_results(batch_status))
                    batch_manager.update_batch(
                        batch_to_update,
                        status=batch_status.status,
                        results=results,
                        errors=errors,
                    )
                    logger.info(
                        f"Batch {batch_to_update} completed with {len(results)} results"
                    )
                else:
                    # Just update status
                    batch_manager.update_batch(
                        batch_to_update, status=batch_status.status
                    )
                    logger.info(
                        f"Updated batch {batch_to_update} status: {batch_status.status}"
                    )

                # Update session state
                st.session_state.batch_jobs[batch_to_update] = batch_manager.get_batch(
                    batch_to_update
                )

                # Rerun the app after poll_interval_seconds
                time.sleep(
                    poll_interval_seconds / 2
                )  # Sleep for half the interval to avoid too frequent reruns
                st.rerun()
            except Exception as e:
                logger.error(f"Error polling batch {batch_to_update}: {str(e)}")
                # Still rerun to try next batch
                time.sleep(poll_interval_seconds)
                st.rerun()

# Tab 3: Results
with tab3:
    st.header("Batch Results")

    # Select batch to view results for
    batch_ids = list(st.session_state.batch_jobs.keys())
    if batch_ids:
        selected_batch = st.selectbox(
            "Select Batch to View",
            options=batch_ids,
            format_func=lambda x: f"{x} ({st.session_state.batch_jobs[x]['status']})",
        )

        batch_info = st.session_state.batch_jobs[selected_batch]

        # Display batch details
        st.subheader("Batch Details")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Status", batch_info["status"])
        col2.metric("Model", batch_info.get("model", "Unknown"))
        col3.metric("Results", len(batch_info.get("results", [])))
        col4.metric("Errors", len(batch_info.get("errors", [])))

        # Results section
        if batch_info.get("results"):
            st.subheader("Results")

            # Convert to dataframe for easier display
            df = format_batch_results(batch_info["results"])

            # Token usage stats
            if len(df) > 0:
                st.subheader("Token Usage")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Input Tokens", df["input_tokens"].sum())
                col2.metric("Total Output Tokens", df["output_tokens"].sum())
                col3.metric("Total Reasoning Tokens", df["reasoning_tokens"].sum())
                col4.metric("Total Tokens", df["total_tokens"].sum())

                # Cost estimate
                cost = calculate_cost_estimate(df)
                st.info(
                    f"Estimated cost: ${cost['total']:.4f} (Input: ${cost['input']:.4f}, Output: ${cost['output']:.4f})"
                )

                # Token usage visualization
                fig = px.bar(
                    pd.DataFrame(
                        {
                            "Token Type": ["Input", "Output", "Reasoning"],
                            "Count": [
                                df["input_tokens"].sum(),
                                df["output_tokens"].sum(),
                                df["reasoning_tokens"].sum(),
                            ],
                        }
                    ),
                    x="Token Type",
                    y="Count",
                    title="Token Usage Breakdown",
                    color="Token Type",
                )
                st.plotly_chart(fig, use_container_width=True)

            # Results table
            with st.expander("View Results Table", expanded=True):
                st.dataframe(
                    df[
                        [
                            "custom_id",
                            "content",
                            "input_tokens",
                            "output_tokens",
                            "reasoning_tokens",
                            "total_tokens",
                        ]
                    ],
                    use_container_width=True,
                )

            # Download buttons
            col1, col2 = st.columns(2)
            with col1:
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Results as CSV",
                    csv,
                    f"batch_{selected_batch}_results.csv",
                    "text/csv",
                    key="download-csv",
                )

            with col2:
                json_str = df.to_json(orient="records")
                st.download_button(
                    "Download Results as JSON",
                    json_str,
                    f"batch_{selected_batch}_results.json",
                    "application/json",
                    key="download-json",
                )

        # Error section
        if batch_info.get("errors"):
            st.subheader("Errors")
            error_data = []
            for error in batch_info["errors"]:
                error_data.append(
                    {
                        "Custom ID": error.get("custom_id", "Unknown"),
                        "Error Type": error.get("error", {}).get("type", "Unknown"),
                        "Error Message": error.get("error", {}).get(
                            "message", "Unknown"
                        ),
                    }
                )

            error_df = pd.DataFrame(error_data)
            st.dataframe(error_df, use_container_width=True)
    else:
        st.info("No batches available to view results for. Submit a batch first.")

# Tab 4: Statistics
with tab4:
    st.header("Batch Processing Statistics")

    # Get batch statistics
    batch_stats = batch_manager.get_stats()

    # Display high-level metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Batches", batch_stats["total_batches"])
    col2.metric("In Progress", batch_stats["in_progress"])
    col3.metric("Completed", batch_stats["completed"])
    col4.metric("Failed", batch_stats["failed"])

    # Batch Status Distribution
    st.subheader("Batch Status Distribution")
    status_data = {
        "Status": ["In Progress", "Completed", "Failed", "Other"],
        "Count": [
            batch_stats["in_progress"],
            batch_stats["completed"],
            batch_stats["failed"],
            batch_stats["other_status"],
        ],
    }

    if sum(status_data["Count"]) > 0:
        fig = px.pie(
            pd.DataFrame(status_data),
            names="Status",
            values="Count",
            color="Status",
            color_discrete_map={
                "In Progress": "#FAA61A",
                "Completed": "#43B581",
                "Failed": "#F04747",
                "Other": "#7289DA",
            },
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No batch data to display yet. Submit some batches first.")

    # Token Usage Statistics (if there are completed batches)
    if batch_stats["completed"] > 0:
        st.subheader("Token Usage Analysis")

        # Collect token usage data from all completed batches
        token_data = {"Input": 0, "Output": 0, "Reasoning": 0}
        batch_models = {}
        total_requests = 0

        for batch_id in batch_manager.get_all_batch_ids():
            batch = batch_manager.get_batch(batch_id)
            if batch and batch["status"] == "completed" and batch.get("results"):
                # Count token usage from results
                batch_df = format_batch_results(batch["results"])
                token_data["Input"] += batch_df["input_tokens"].sum()
                token_data["Output"] += batch_df["output_tokens"].sum()
                token_data["Reasoning"] += batch_df["reasoning_tokens"].sum()

                # Count requests by model
                model = batch.get("model", "Unknown")
                batch_models[model] = batch_models.get(model, 0) + len(batch["results"])
                total_requests += len(batch["results"])

        # Show token usage breakdown
        col1, col2 = st.columns(2)

        with col1:
            # Token usage by type
            if sum(token_data.values()) > 0:
                fig1 = px.bar(
                    pd.DataFrame(
                        {
                            "Token Type": list(token_data.keys()),
                            "Count": list(token_data.values()),
                        }
                    ),
                    x="Token Type",
                    y="Count",
                    color="Token Type",
                    title="Token Usage by Type",
                )
                st.plotly_chart(fig1, use_container_width=True)

        with col2:
            # Model distribution
            if batch_models:
                fig2 = px.pie(
                    pd.DataFrame(
                        {
                            "Model": list(batch_models.keys()),
                            "Requests": list(batch_models.values()),
                        }
                    ),
                    names="Model",
                    values="Requests",
                    title="Requests by Model",
                )
                st.plotly_chart(fig2, use_container_width=True)

        # Cost estimation
        if sum(token_data.values()) > 0:
            st.subheader("Cost Estimation")

            # Estimate costs based on token usage
            total_tokens = sum(token_data.values())
            # Create a dataframe with token usage from all models
            model_df = pd.DataFrame(
                {
                    "model": list(batch_models.keys()),
                    "input_tokens": [
                        token_data["Input"] * (batch_models[m] / total_requests)
                        for m in batch_models.keys()
                    ],
                    "output_tokens": [
                        token_data["Output"] * (batch_models[m] / total_requests)
                        for m in batch_models.keys()
                    ],
                    "reasoning_tokens": [
                        token_data["Reasoning"] * (batch_models[m] / total_requests)
                        for m in batch_models.keys()
                    ],
                }
            )

            cost = calculate_cost_estimate(model_df)

            col1, col2, col3 = st.columns(3)
            col1.metric("Input Cost", f"${cost['input']:.4f}")
            col2.metric("Output Cost", f"${cost['output']:.4f}")
            col3.metric("Total Cost", f"${cost['total']:.4f}")

            # Cost over time (if we had timestamps)
            st.info(
                "💡 Pro tip: This is an estimated cost based on current pricing. Actual billing may vary."
            )

    # Batch Management
    st.subheader("Batch Management")

    col1, col2 = st.columns(2)

    with col1:
        # Prune old batches
        prune_days = st.number_input(
            "Prune batches older than (days)", min_value=1, value=30
        )
        if st.button("Prune Old Batches"):
            with st.spinner(f"Pruning batches older than {prune_days} days..."):
                pruned_count = batch_manager.prune_old_batches(prune_days)
                if pruned_count > 0:
                    st.success(f"Pruned {pruned_count} old batches")
                else:
                    st.info("No batches needed pruning")

    with col2:
        # Batch storage info
        if os.path.exists(batch_manager.storage_dir):
            batch_files = glob.glob(os.path.join(batch_manager.storage_dir, "*.json"))
            total_size = sum(os.path.getsize(f) for f in batch_files)

            st.metric("Storage Files", len(batch_files))
            st.metric("Storage Size", f"{total_size / (1024*1024):.2f} MB")

# Tab 5: Logs
with tab5:
    st.header("Logs")

    toggle_pressed = st.button(
        "Hide Logs" if st.session_state.show_logs else "Show Logs",
        key="toggle-logs",
    )
    if toggle_pressed:
        st.session_state.show_logs = not st.session_state.show_logs

    if st.session_state.show_logs and st.button("Get Logs"):
        logs = get_recent_logs()
        st.code("\n".join(logs), language="text")

# Footer
st.markdown("---")
st.markdown(
    "**OpenAI o1-Pro Batch API Interface** | Built with Streamlit",
    unsafe_allow_html=True,
)
