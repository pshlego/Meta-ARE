from __future__ import annotations
from typing import Literal

"""
Available models for CloudGPT OpenAI
"""
cloudgpt_available_models = Literal[
    ##########################################################################
    # OpenAI Models
    ##########################################################################
    # Current OpenAI models
    "gpt-4o-20240513",  # ChatCompletions, Responses
    "gpt-4o-20240806",  # ChatCompletions, Responses
    "gpt-4o-20241120",  # ChatCompletions, Responses
    "gpt-4o-mini-20240718",  # ChatCompletions, Responses
    "gpt-4.1-20250414",  # ChatCompletions, Responses
    "gpt-4.1-mini-20250414",  # ChatCompletions, Responses
    "gpt-4.1-nano-20250414",  # ChatCompletions, Responses
    "gpt-5-20250807",  # ChatCompletions, Responses
    "gpt-5-mini-20250807",  # ChatCompletions, Responses
    "gpt-5-nano-20250807",  # ChatCompletions, Responses
    "gpt-5-pro-20251006",  # ChatCompletions, Responses
    "gpt-5.1-20251113",  # ChatCompletions, Responses
    "gpt-5.2-20251211",  # ChatCompletions, Responses
    "gpt-5.4-20260305",  # ChatCompletions, Responses
    "gpt-5.4-pro-20260305",  # Responses
    "gpt-5.4-mini-20260317",  # ChatCompletions, Responses
    "gpt-5.4-nano-20260317",  # ChatCompletions, Responses
    "gpt-5.5-20260424",  # ChatCompletions, Responses
    # Chat models
    "gpt-5-chat-20250807",  # ChatCompletions, Responses
    "gpt-5-chat-20251003",  # ChatCompletions, Responses
    "gpt-5.1-chat-20251113",  # ChatCompletions, Responses
    "gpt-5.2-chat-20251211",  # ChatCompletions, Responses
    "gpt-5.2-chat-20260210",  # ChatCompletions, Responses
    "gpt-5.3-chat-20260303",  # ChatCompletions, Responses
    "gpt-chat-latest-20260505",  # ChatCompletions, Responses
    # Coding models
    "codex-mini-20250516",  # Responses
    "gpt-5-codex-20250915",  # Responses
    "gpt-5.1-codex-20251113",  # Responses
    "gpt-5.1-codex-mini-20251113",  # Responses
    "gpt-5.1-codex-max-20251204",  # Responses
    "gpt-5.2-codex-20260114",  # Responses
    "gpt-5.3-codex-20260224",  # Responses
    # Computer use models
    "computer-use-preview-20250311",  # Responses
    # Reasoning models
    "o1-20241217", # ChatCompletions, Responses
    "o3-mini-20250131", # ChatCompletions, Responses
    "o3-20250416",  # ChatCompletions, Responses
    "o3-pro-20250610",  # Responses
    "o3-deep-research-20250626",  # Responses
    "o4-mini-20250416",  # ChatCompletions, Responses
    # Embedding models
    "text-embedding-ada-002", # Embeddings
    "text-embedding-3-small", # Embeddings
    "text-embedding-3-large", # Embeddings
    # Open source models
    "gpt-oss-20b", # ChatCompletions
    "gpt-oss-120b", # ChatCompletions
    ##########################################################################
    # xAI Models
    ##########################################################################
    "grok-3", # ChatCompletions
    "grok-3-mini", # ChatCompletions
    "grok-4", # ChatCompletions
    "grok-4-fast-reasoning", # ChatCompletions
    "grok-4-fast-non-reasoning", # ChatCompletions
    "grok-code-fast-1", # ChatCompletions
    "grok-4-1-fast-reasoning", # ChatCompletions
    "grok-4-1-fast-non-reasoning", # ChatCompletions
    "grok-4-20-reasoning", # ChatCompletions
    "grok-4-20-non-reasoning", # ChatCompletions
    "grok-4.3", # ChatCompletions
    ##########################################################################
    # DeepSeek Models
    ##########################################################################
    "DeepSeek-V3-0324", # ChatCompletions
    "DeepSeek-R1", # ChatCompletions
    "DeepSeek-R1-0528", # ChatCompletions
    "DeepSeek-V3.1", # ChatCompletions
    "DeepSeek-V3.2", # ChatCompletions
    "DeepSeek-V3.2-Speciale", # ChatCompletions
    "DeepSeek-V4-Flash", # ChatCompletions
    "DeepSeek-V4-Pro", # ChatCompletions
    ##########################################################################
    # Moonshot Models
    ##########################################################################
    "Kimi-K2-Thinking", # ChatCompletions
    "Kimi-K2.5", # ChatCompletions
    "Kimi-K2.6", # ChatCompletions
    ##########################################################################
    # Meta Models
    ##########################################################################
    "Llama-3.3-70B-Instruct", # ChatCompletions
    "Llama-4-Maverick-17B-128E-Instruct-FP8", # ChatCompletions
    ##########################################################################
    # Image Generation Models
    ##########################################################################
    "gpt-image-1", # ImageGeneration, ImageEdit
    "gpt-image-1-mini", # ImageGeneration, ImageEdit
    "gpt-image-1.5", # ImageGeneration, ImageEdit
    "gpt-image-2", # ImageGeneration, ImageEdit
    "MAI-Image-2", # MAI-ImageGeneration
    "MAI-Image-2e", # MAI-ImageGeneration
    ##########################################################################
    # Video Generation Models
    ##########################################################################
    "sora-20250502", # Videos
    "sora-2-20251006", # Videos
]

###########################################################################
# ⚠️ Deprecated Models
##########################################################################
# gpt-35-turbo-20220309: deprecated, redirected to gpt-4.1-mini-20250414
# gpt-35-turbo-16k-20230613: deprecated, redirected to gpt-4.1-mini-20250414
# gpt-35-turbo-20230613: deprecated, redirected to gpt-4.1-mini-20250414
# gpt-4-1106-preview: deprecated, redirected to gpt-4o-20241120
# gpt-4-0125-preview: deprecated, redirected to gpt-4o-20241120
# gpt-4-20230321: deprecated, redirected to gpt-4o-20241120
# gpt-4-20230613: deprecated, redirected to gpt-4o-20241120
# gpt-4-32k-20230321: deprecated, redirected to gpt-4o-20241120
# gpt-4-32k-20230613: deprecated, redirected to gpt-4o-20241120
# gpt-4-visual-preview: deprecated, redirected to gpt-4o-20241120
# gpt-4o-audio-preview-20241217: deprecated
# gpt-4.5-preview-20250227: deprecated
# deepseek-r1-preview: deprecated, redirected to DeepSeek-R1
# gpt-5-chat-20250807: deprecated
# gpt-4o-realtime-preview-20241001: deprecated, realtime models preview support has been removed
# gpt-35-turbo-1106: deprecated
# gpt-35-turbo-0125: deprecated
# gpt-4-turbo-20240409: deprecated
# o1-mini-20240912: deprecated, redirected to o4-mini-20250416

from typing import (
    Any,
    Callable,
    Coroutine,
    Literal,
    Optional,
    ParamSpec,
    TypeVar,
    cast,
    Dict,
    TYPE_CHECKING,
    overload,
)
import sys, os
import functools

__all__ = [
    "get_openai_token_provider",
    "get_openai_token",
    "get_openai_client",
    "get_chat_completion",
    "encode_image",
    "cloudgpt_available_models",
]

TokenProvider = Callable[[], str]
AsyncTokenProvider = Callable[[], Coroutine[Any, Any, str]]


def check_module():
    try:
        import openai, azure.identity.broker  # type: ignore

        del openai, azure.identity.broker
    except ImportError:
        print("Please install the required packages by running the following command:")
        print("pip install openai azure-identity-broker --upgrade")
        exit(1)


check_module()

import openai
from openai import OpenAI

_depRt = TypeVar("_depRt")
_depParam = ParamSpec("_depParam")


def _deprecated(message: str):
    def deprecated_decorator(
        func: Callable[_depParam, _depRt],
    ) -> Callable[_depParam, _depRt]:
        def deprecated_func(
            *args: _depParam.args, **kwargs: _depParam.kwargs
        ) -> _depRt:
            import traceback

            print(
                "\n ⚠️  \x1b[31m{} is a deprecated function. {}".format(
                    func.__name__, message
                )
            )
            traceback.print_stack()
            print("\x1b[0m")
            return func(*args, **kwargs)

        return deprecated_func

    return deprecated_decorator


def _validate_token(token: str) -> bool:
    import requests

    url = "https://cloudgpt-openai.azure-api.net/openai/ping"

    headers = {
        "Authorization": f"Bearer {token}",
    }
    try:
        response = requests.get(url, headers=headers)
        assert response.status_code == 200 and response.text == "OK", response.text
        return True
    except Exception as e:
        print("Failed to validate token", e)
        return False


@functools.lru_cache(maxsize=3)
def get_openai_token_provider(
    token_cache_file: str = "cloudgpt-apim-token-cache.bin",
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    use_azure_cli: Optional[bool] = None,
    use_broker_login: Optional[bool] = None,
    use_managed_identity: Optional[bool] = None,
    use_device_code: Optional[bool] = None,
    skip_access_validation: Optional[bool] = False,
    **kwargs: Any,
) -> TokenProvider:
    """
    Get a token provider function that could return a valid access token for CloudGPT OpenAI.

    The return value is a function that should be used with AzureOpenAIClient constructor as azure_ad_token_provider parameter.
    The following code snippet shows how to use it with AzureOpenAIClient:

    ```python
    token_provider = get_openai_token_provider()
    client = openai.AzureOpenAI(
        api_version="2025-03-01-preview",
        azure_endpoint="https://cloudgpt-openai.azure-api.net/",
        azure_ad_token_provider=token_provider,
    )
    ```

    Parameters
    ----------
    token_cache_file : str, optional
        path to the token cache file, by default 'cloudgpt-apim-token-cache.bin' in the current directory
    client_id : Optional[str], optional
        client id for AAD app, by default None
    client_secret : Optional[str], optional
        client secret for AAD app, by default None
    use_azure_cli : Optional[bool], optional
        use Azure CLI for authentication, by default None. If AzCli has been installed and logged in,
        it will be used for authentication. This is recommended for headless environments and AzCLI takes
        care of token cache and token refresh.
    use_broker_login : Optional[bool], optional
        use broker login for authentication, by default None.
        If not specified, it will be enabled for known supported environments (e.g. Windows, macOS, WSL, VSCode),
        but sometimes it may not always could cache the token for long-term usage.
        In such cases, you can disable it by setting it to False.
    use_managed_identity : Optional[bool], optional
        use managed identity for authentication, by default None.
        If not specified, it will use user assigned managed identity if client_id is specified,
        For use system assigned managed identity, client_id could be None but need to set use_managed_identity to True.
    use_device_code : Optional[bool], optional
        use device code for authentication, by default None. If not specified, it will use interactive login on supported platform.
    skip_access_validation : Optional[bool], optional
        skip access token validation, by default False.

    Returns
    -------
    TokenProvider
        the token provider function that could return a valid access token for CloudGPT OpenAI
    """
    import shutil
    from azure.identity.broker import InteractiveBrowserBrokerCredential
    from azure.identity import (
        ManagedIdentityCredential,
        ClientSecretCredential,
        DeviceCodeCredential,
        AuthenticationRecord,
        AzureCliCredential,
    )
    from azure.identity import TokenCachePersistenceOptions
    import msal  # type: ignore

    api_scope_base = "api://feb7b661-cac7-44a8-8dc1-163b63c23df2"
    tenant_id = "72f988bf-86f1-41af-91ab-2d7cd011db47"
    scope = api_scope_base + "/.default"

    token_cache_option = TokenCachePersistenceOptions(
        name=token_cache_file,
        enable_persistence=True,
        allow_unencrypted_storage=True,
    )

    def save_auth_record(auth_record: AuthenticationRecord):
        try:
            with open(token_cache_file, "w") as cache_file:
                cache_file.write(auth_record.serialize())
        except Exception as e:
            print("failed to save auth record", e)

    def load_auth_record() -> Optional[AuthenticationRecord]:
        try:
            if not os.path.exists(token_cache_file):
                return None
            with open(token_cache_file, "r") as cache_file:
                return AuthenticationRecord.deserialize(cache_file.read())
        except Exception as e:
            print("failed to load auth record", e)
            return None

    auth_record: Optional[AuthenticationRecord] = load_auth_record()

    current_auth_mode: Literal[
        "client_secret",
        "managed_identity",
        "az_cli",
        "interactive",
        "device_code",
        "none",
    ] = "none"

    implicit_mode = not (
        use_managed_identity or use_azure_cli or use_broker_login or use_device_code
    )

    if use_managed_identity or (implicit_mode and client_id is not None):
        if not use_managed_identity and client_secret is not None:
            assert (
                client_id is not None
            ), "client_id must be specified with client_secret"
            current_auth_mode = "client_secret"
            identity = ClientSecretCredential(
                client_id=client_id,
                client_secret=client_secret,
                tenant_id=tenant_id,
                cache_persistence_options=token_cache_option,
                authentication_record=auth_record,
            )
        else:
            current_auth_mode = "managed_identity"
            if client_id is None:
                # using default managed identity
                identity = ManagedIdentityCredential(
                    cache_persistence_options=token_cache_option,
                )
            else:
                identity = ManagedIdentityCredential(
                    client_id=client_id,
                    cache_persistence_options=token_cache_option,
                )
    elif use_azure_cli or (implicit_mode and shutil.which("az") is not None):
        current_auth_mode = "az_cli"
        identity = AzureCliCredential(tenant_id=tenant_id)
    else:
        if implicit_mode:
            # enable broker login for known supported envs if not specified using use_device_code
            if sys.platform.startswith("darwin") or sys.platform.startswith("win32"):
                use_broker_login = True
            elif os.environ.get("WSL_DISTRO_NAME", "") != "":
                use_broker_login = True
            elif os.environ.get("TERM_PROGRAM", "") == "vscode":
                use_broker_login = True
            else:
                use_broker_login = False
        if use_broker_login:
            current_auth_mode = "interactive"
            identity = InteractiveBrowserBrokerCredential(
                tenant_id="72f988bf-86f1-41af-91ab-2d7cd011db47",
                cache_persistence_options=token_cache_option,
                use_default_broker_account=True,
                parent_window_handle=msal.PublicClientApplication.CONSOLE_WINDOW_HANDLE,
                authentication_record=auth_record,
            )
        else:
            current_auth_mode = "device_code"
            identity = DeviceCodeCredential(
                tenant_id="72f988bf-86f1-41af-91ab-2d7cd011db47",
                cache_persistence_options=token_cache_option,
                authentication_record=auth_record,
            )

        try:
            auth_record = identity.authenticate(scopes=[scope])
            if auth_record:
                save_auth_record(auth_record)

        except Exception as e:
            print(
                f"failed to acquire token from AAD for CloudGPT OpenAI using {current_auth_mode}",
                e,
            )
            raise e

    try:
        from azure.identity import get_bearer_token_provider

        token_provider = get_bearer_token_provider(identity, scope)
        token_verified_cache: str = ""

        def token_provider_wrapper():
            nonlocal token_verified_cache
            token = token_provider()
            if token != token_verified_cache:
                if not skip_access_validation:
                    assert _validate_token(token), "failed to validate token"
                token_verified_cache = token
            return token

        return token_provider_wrapper
    except Exception as e:
        print("failed to acquire token from AAD for CloudGPT OpenAI", e)
        raise e


@functools.lru_cache(maxsize=3)
async def async_get_openai_token_provider(
    **kwargs: Any,
) -> AsyncTokenProvider:
    # TODO: implement async version of get_openai_token_provider
    token_provider = get_openai_token_provider(
        **kwargs,
    )

    async def async_token_provider() -> str:
        return token_provider()

    return async_token_provider


@_deprecated(
    "use get_openai_token_provider instead whenever possible "
    "and use it as the azure_ad_token_provider parameter in AzureOpenAIClient constructor. "
    "Please do not acquire token directly or use it elsewhere."
)
def get_openai_token(
    token_cache_file: str = "cloudgpt-apim-token-cache.bin",
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    use_azure_cli: Optional[bool] = None,
    use_broker_login: Optional[bool] = None,
    use_managed_identity: Optional[bool] = None,
    use_device_code: Optional[bool] = None,
    skip_access_validation: Optional[bool] = False,
    **kwargs: Any,
) -> str:
    """
    get access token for CloudGPT OpenAI
    """
    return get_openai_token_provider(
        token_cache_file=token_cache_file,
        client_id=client_id,
        client_secret=client_secret,
        use_azure_cli=use_azure_cli,
        use_broker_login=use_broker_login,
        use_managed_identity=use_managed_identity,
        use_device_code=use_device_code,
        skip_access_validation=skip_access_validation,
        **kwargs,
    )()


def encode_image(image_path: str, mime_type: Optional[str] = None) -> str:
    """
    Utility function to encode image to base64 for using in OpenAI API

    Parameters
    ----------
    image_path : str
        path to the image file

    mime_type : Optional[str], optional
        mime type of the image, by default None and will infer from the file extension if possible

    Returns
    -------
    str
        base64 encoded image url
    """
    import base64
    import mimetypes

    file_name = os.path.basename(image_path)
    mime_type = cast(
        Optional[str],
        mime_type if mime_type is not None else mimetypes.guess_type(file_name)[0],  # type: ignore
    )
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("ascii")

    if mime_type is None or not mime_type.startswith("image/"):
        print(
            "Warning: mime_type is not specified or not an image mime type. Defaulting to png."
        )
        mime_type = "image/png"

    image_url = f"data:{mime_type};base64," + encoded_image
    return image_url


@functools.lru_cache(maxsize=3)
def get_openai_client(
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    use_azure_cli: Optional[bool] = None,
    use_broker_login: Optional[bool] = None,
    use_managed_identity: Optional[bool] = None,
    use_device_code: Optional[bool] = None,
    v1_api: Optional[bool] = None,
    default_headers: Optional[Dict[str, str]] = None,
) -> OpenAI:
    """
    Initialize OpenAI client for CloudGPT OpenAI.

    All parameters are optional and will use the default authentication method if not specified.

    Parameters
    ----------
    client_id : Optional[str], optional
        client id for AAD app, by default None
    client_secret : Optional[str], optional
        client secret for AAD app, by default None
    use_azure_cli : Optional[bool], optional
        use Azure CLI for authentication, by default None. If AzCli has been installed and logged in,
        it will be used for authentication. This is recommended for headless environments and AzCLI takes
        care of token cache and token refresh.
    use_broker_login : Optional[bool], optional
        use broker login for authentication, by default None.
        If not specified, it will be enabled for known supported environments (e.g. Windows, macOS, WSL, VSCode),
        but sometimes it may not always could cache the token for long-term usage.
        In such cases, you can disable it by setting it to False.
    use_managed_identity : Optional[bool], optional
        use managed identity for authentication, by default None.
        If not specified, it will use user assigned managed identity if client_id is specified,
        For use system assigned managed identity, client_id could be None but need to set use_managed_identity to True.
    use_device_code : Optional[bool], optional
        use device code for authentication, by default None. If not specified, it will use interactive login on supported platform.

    Returns
    -------
    OpenAI
        OpenAI client for CloudGPT OpenAI. Check https://github.com/openai/openai-python for more details.
    """
    token_provider = get_openai_token_provider(
        client_id=client_id,
        client_secret=client_secret,
        use_azure_cli=use_azure_cli,
        use_broker_login=use_broker_login,
        use_managed_identity=use_managed_identity,
        use_device_code=use_device_code,
    )
    token_provider()
    if v1_api:
        api_version = "preview"
        base_url = "https://cloudgpt-openai.azure-api.net/openai/v1/"
    else:
        api_version = "2025-04-01-preview"
        base_url = "https://cloudgpt-openai.azure-api.net/openai/"
    client = openai.AzureOpenAI(
        api_version=api_version,
        base_url=base_url,
        azure_ad_token_provider=token_provider,
        default_headers=default_headers,
    )
    return client


@functools.lru_cache(maxsize=3)
async def async_get_openai_client(
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    use_azure_cli: Optional[bool] = None,
    use_broker_login: Optional[bool] = None,
    use_managed_identity: Optional[bool] = None,
    use_device_code: Optional[bool] = None,
    v1_api: Optional[bool] = None,
    default_headers: Optional[Dict[str, str]] = None,
) -> openai.AsyncOpenAI:
    token_provider = await async_get_openai_token_provider(
        client_id=client_id,
        client_secret=client_secret,
        use_azure_cli=use_azure_cli,
        use_broker_login=use_broker_login,
        use_managed_identity=use_managed_identity,
        use_device_code=use_device_code,
    )
    await token_provider()
    if v1_api:
        api_version = "preview"
        base_url = "https://cloudgpt-openai.azure-api.net/openai/v1/"
    else:
        api_version = "2025-04-01-preview"
        base_url = "https://cloudgpt-openai.azure-api.net/openai/"
    client = openai.AsyncAzureOpenAI(
        api_version=api_version,
        base_url=base_url,
        azure_ad_token_provider=token_provider,
        default_headers=default_headers,
    )
    return client


if TYPE_CHECKING:
    from openai.types.chat.chat_completion_message_param import (
        ChatCompletionMessageParam,
    )
    from openai.types.chat.chat_completion import ChatCompletion
    from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
    from typing import Iterable


@overload
def get_chat_completion(
    messages: "Iterable[ChatCompletionMessageParam]",
    model: cloudgpt_available_models,
    stream: "Literal[False]" = False,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    use_azure_cli: Optional[bool] = None,
    use_broker_login: Optional[bool] = None,
    use_managed_identity: Optional[bool] = None,
    use_device_code: Optional[bool] = None,
    v1_api: Optional[bool] = None,
    **kwargs: Any,
) -> "ChatCompletion": ...
@overload
def get_chat_completion(
    messages: "Iterable[ChatCompletionMessageParam]",
    model: cloudgpt_available_models,
    stream: Literal[True],
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    use_azure_cli: Optional[bool] = None,
    use_broker_login: Optional[bool] = None,
    use_managed_identity: Optional[bool] = None,
    use_device_code: Optional[bool] = None,
    v1_api: Optional[bool] = None,
    **kwargs: Any,
) -> "openai.Stream[ChatCompletionChunk]": ...
def get_chat_completion(
    messages: "Iterable[ChatCompletionMessageParam]",
    model: Optional[cloudgpt_available_models] = None,
    stream: bool = False,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    use_azure_cli: Optional[bool] = None,
    use_broker_login: Optional[bool] = None,
    use_managed_identity: Optional[bool] = None,
    use_device_code: Optional[bool] = None,
    v1_api: Optional[bool] = None,
    **kwargs: Any,
) -> "ChatCompletion | openai.Stream[ChatCompletionChunk]":
    """
    Helper function to get chat completion from OpenAI API
    """

    engine: Optional[str] = kwargs.get("engine")

    model_name: Any = model
    if model_name is None:
        if engine is None:
            raise ValueError("model name must be specified by 'model' parameter")
        model_name = engine

    if "engine" in kwargs:
        del kwargs["engine"]

    client = get_openai_client(
        client_id=client_id,
        client_secret=client_secret,
        use_azure_cli=use_azure_cli,
        use_broker_login=use_broker_login,
        use_managed_identity=use_managed_identity,
        use_device_code=use_device_code,
        v1_api=v1_api,
    )

    response = client.chat.completions.create(
        messages=messages, model=model_name, stream=stream, **kwargs
    )

    return response


@overload
async def async_get_chat_completion(
    messages: "Iterable[ChatCompletionMessageParam]",
    model: cloudgpt_available_models,
    stream: "Literal[False]" = False,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    use_azure_cli: Optional[bool] = None,
    use_broker_login: Optional[bool] = None,
    use_managed_identity: Optional[bool] = None,
    use_device_code: Optional[bool] = None,
    v1_api: Optional[bool] = None,
    **kwargs: Any,
) -> "ChatCompletion": ...
@overload
async def async_get_chat_completion(
    messages: "Iterable[ChatCompletionMessageParam]",
    model: cloudgpt_available_models,
    stream: "Literal[True]",
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    use_azure_cli: Optional[bool] = None,
    use_broker_login: Optional[bool] = None,
    use_managed_identity: Optional[bool] = None,
    use_device_code: Optional[bool] = None,
    v1_api: Optional[bool] = None,
    **kwargs: Any,
) -> "openai.AsyncStream[ChatCompletionChunk]": ...
async def async_get_chat_completion(
    messages: Iterable[ChatCompletionMessageParam],
    model: Optional[cloudgpt_available_models] = None,
    stream: bool = False,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    use_azure_cli: Optional[bool] = None,
    use_broker_login: Optional[bool] = None,
    use_managed_identity: Optional[bool] = None,
    use_device_code: Optional[bool] = None,
    v1_api: Optional[bool] = None,
    **kwargs: Any,
) -> "ChatCompletion | openai.AsyncStream[ChatCompletionChunk]":
    """
    Helper function to get chat completion from OpenAI API with async API
    """

    engine: Optional[str] = kwargs.get("engine")

    model_name: Any = model
    if model_name is None:
        if engine is None:
            raise ValueError("model name must be specified by 'model' parameter")
        model_name = engine

    if "engine" in kwargs:
        del kwargs["engine"]

    client = await async_get_openai_client(
        client_id=client_id,
        client_secret=client_secret,
        use_azure_cli=use_azure_cli,
        use_broker_login=use_broker_login,
        use_managed_identity=use_managed_identity,
        use_device_code=use_device_code,
        v1_api=v1_api,
    )

    response = await client.chat.completions.create(
        messages=messages, model=model_name, stream=stream, **kwargs
    )

    return response



def _test_call(**kwargs: Any):
    test_message = "What is the content?"

    client = get_openai_client(**kwargs)

    response = client.chat.completions.create(
        model="gpt-4o-mini-20240718",
        messages=[{"role": "user", "content": test_message}],
        temperature=0.7,
        max_tokens=100,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
    )

    print(response.choices[0].message)


if __name__ == "__main__":
    _test_call(use_broker_login=True)
